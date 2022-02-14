library(tidyverse)
library(cmdstanr)
library(posterior)
library(bayesplot)
library(ggfan)
library(cowplot)
initialize_purrr_progress = function(x){
	.pp <<- progress::progress_bar$new(
		format = "[:bar] :percent"# eta: :eta",
		total = length(x), clear = FALSE, width= 60
	)
	return(x)
}
finalize_purrr_progress = function(x){
	x = identity(x) #to break lazy eval in pipe
	.pp$terminate()
	dt = difftime(Sys.time(),.pp$.__enclos_env__$private$start)
	cat('Operations took ',round(dt),' ',attr(dt,'units'))
	return(x)
}

generate_mod = cmdstan_model('stan/pvgp_generate.stan')
sample_mod = cmdstan_model('stan/pvgp_sample.stan')

f = function(z){
	if(!is.null(all_out)){
		if(nrow(semi_join(z,all_out,by=c('n','iter')))==1){
			return(NULL)
		}
	}
	n = z$n
	x = 1:n
	data_for_gen = lst(
		n = n
		, x = 1:n
		, pvn_ = .2
		, lengthscale_ = .1*diff(range(x))
		, sample_pvn = F
		, sample_lengthscale = F
	)
	capture.output(
		generated <- generate_mod$generate_quantities(
			data = data_for_gen
			, fitted_params = as_draws_array(tibble(dummy=0))
		)
	) -> captured_output

	#extract quantities
	(
		generated$draws(c('f','y'),format ="draws_list" )
		%>% as_draws_df()
		%>% as_tibble()
		%>% select(-.chain,-.iteration,-.draw)
		%>% pivot_longer(everything())
		%>% separate(
			name
			, into = c('name','index')
			, convert = TRUE
			, extra = 'drop'
		)
		# %>% View())
		%>% pivot_wider()
		%>% mutate(
			x = x[index]
		)
	) -> dat

	# compute distance matrix
	dx = matrix(NA,n,n)
	for(i in 1:n){
		for(j in 1:n){
			dx[i,j] = ((x[i]-x[j])^2)/-2 # divided by negative two to match standard kernel
		}
	}
	dx_ut_flat = as.numeric(dx[upper.tri(dx)])
	dx_lt_flat = as.numeric(dx[lower.tri(dx)])

	data_for_stan = lst(
		n_xy = nrow(dat)
		, x = dat$x
		, y = dat$y
		# udx: vector of unique differences in x (squared & divided by -2)
		, udx = unique(dx_ut_flat)
		# i_uk_k_ut: index array mapping between unique entries in k to the upper-triangle of k
		, i_uk_k_ut = match(dx_ut_flat,udx)
		# i_uk_k_lt: index array mapping between unique entries in k to the lower-triangle of k
		, i_uk_k_lt = match(dx_lt_flat,udx)
		# i_k_diag: index array indicating elements in flattened covariance matrix that are on the diagonal when unflattened
		, i_k_diag = which(as.numeric(diag(1,n,n))==1)
		# i_k_ut: index array indicating elements in flattened covariance matrix that are in the upper triangle when unflattened
		, i_k_ut = which(as.numeric(upper.tri(matrix(NA,n,n)))==1)
		# i_k_lt: index array indicating elements in flattened covariance matrix that are in the lower triangle when unflattened
		, i_k_lt = which(as.numeric(lower.tri(matrix(NA,n,n)))==1)
		# n_udx: number of unique differences in x
		, n_udx = length(udx)
		# n_ut: number of elements in covariance upper triangle
		, n_ut = length(i_k_ut)
	)

	#sample
	capture.output(
		sampled <- sample_mod$sample(
			data = data_for_stan
			, chains = parallel::detectCores()/2
			, parallel_chains = parallel::detectCores()/2
			, refresh = 0
			, show_messages = F
		)
	) -> captured_output

	#get max delta
	(
		sampled$draws('delta',format='draws_list')
		%>% map(.f=max)
		%>% unlist()
	) -> delta

	#get timing
	p = sampled$profiles()
	(
		p
		%>% bind_rows(.id = 'chain')
		%>% rename(value=total_time)
		%>% select(chain,name,value)
		%>% pivot_wider()
		%>% bind_cols(z)
		%>% mutate( delta = delta )
	) ->
		this_out
	all_out <<-bind_rows(all_out,this_out)
	saveRDS(all_out,file='all_out.rds')
	do_plots()
	# print(plot_grid(p1,p2,p3,nrow=3))#,rel_widths=c(5,1,5,1,5)))
	return(NULL)
}

do_plots = function(){
	(
		all_out
		%>% select(eqc,udx,n)
		%>% pivot_longer(cols=c(eqc,udx))
		%>% group_by(name,n)
		%>% summarize(
			med = median(value)
			, lo80 = quantile(value,.1)
			, hi80 = quantile(value,.9)
			, lo50 = quantile(value,.25)
			, hi50 = quantile(value,.75)
			, .groups = 'drop'
		)
		%>% ggplot()
		+ geom_line(aes(x=n,y=med,color=name))
		+ geom_linerange(aes(x=n,ymin=lo50,ymax=hi50,color=name),size=3,alpha=.5)
		+ geom_linerange(aes(x=n,ymin=lo80,ymax=hi80,color=name),size=1,alpha=.5)
		+ scale_y_log10(name='Time (s)')
		+ scale_x_log10(name= '# grid points in 1D GP')
		+ labs(color = 'Method')
		+ theme(legend.position = 'top')
	) -> p1
	(
		all_out
		%>% mutate(value = eqc/udx)
		%>% group_by(n)
		%>% summarize(
			med = median(value)
			, lo80 = quantile(value,.1)
			, hi80 = quantile(value,.9)
			, lo50 = quantile(value,.25)
			, hi50 = quantile(value,.75)
			, delta = max(delta)
			, .groups = 'drop'
		)
		%>% ggplot()
		+ geom_hline(yintercept = 1, linetype=3)
		+ geom_line(aes(x=n,y=med))
		+ geom_linerange(aes(x=n,ymin=lo50,ymax=hi50),size=3)
		+ geom_linerange(aes(x=n,ymin=lo80,ymax=hi80),size=1)
		# + geom_errorbar(aes(x=n,ymin=lo80,ymax=hi80,color=log(delta)),width=.1)
		# + geom_point(aes(x=n,y=med,color=log(delta)),size=3)
		# + scale_color_gradient(name='log(max(Δ))\n')
		+ scale_y_continuous(name='Time Ratio')
		+ scale_x_log10(name= '# grid points in 1D GP')
		+ theme(legend.position = 'top')
	) -> p2

	(
		all_out
		%>% mutate(
			value = (eqc+f_+model+tp)/(udx+f_+model+tp)
		)
		%>% group_by(n)
		%>% summarize(
			med = median(value)
			, lo80 = quantile(value,.1)
			, hi80 = quantile(value,.9)
			, lo50 = quantile(value,.25)
			, hi50 = quantile(value,.75)
			, delta = max(delta)
			, .groups = 'drop'
		)
		%>% ggplot()
		+ geom_hline(yintercept = 1, linetype=3)
		+ geom_line(aes(x=n,y=med))
		+ geom_linerange(aes(x=n,ymin=lo50,ymax=hi50),size=3)
		+ geom_linerange(aes(x=n,ymin=lo80,ymax=hi80),size=1)
		+ scale_y_continuous(name='Total (inc. rest of model) Time Ratio')
		+ scale_x_log10(name= '# grid points in 1D GP')
		+ theme(legend.position = 'top')
	) -> p3
	print(plot_grid(p1,NULL,p2,NULL,p3,nrow=1,rel_widths=c(5,1,5,1,5)))
	return(NULL)
}
f_tick = function(z){
	f(z)
	.pp$tick()
	return(z)
}

if(file.exists('all_out.rds')){
	all_out = readRDS('all_out.rds')
	do_plots()
}else{
	all_out = NULL
}
#all_out = NULL
(
	expand_grid(
		n = 2^8#2^(2:10)
		, iter = 1:1e2
	)
	%>% group_by(n,iter)
	%>% group_split()
	%>% initialize_purrr_progress()
	%>% walk(
		.f = f_tick
	)
	%>% finalize_purrr_progress()
)

