library(tidyverse)
library(cmdstanr)
library(posterior)
library(bayesplot)
library(ggfan)

#generate some data
n = 2^5 #bigger n will take longer; 2^7 takes about 90s on a reasonably modern cpu
#x can be anything (doesn't have to be on a grid)
x = 1:n
# x = sort(runif(n,0,1))
generate_mod = cmdstan_model('pvgp_generate.stan',force_recompile = T)
generated = generate_mod$generate_quantities(
	data = lst(
		n = n
		, x = x
		, pvn_ = .2
		, lengthscale_ = .1*diff(range(x))
		, sample_pvn = F
		, sample_lengthscale = F
	)
	, fitted_params = as_draws_array(tibble(dummy=0))
)

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
	)
	# %>% View())
	%>% pivot_wider()
	%>% mutate(
		x = x[index]
	)
) -> dat

#quick viz:
(
	dat
	%>% ggplot(aes(x=x))
	+ geom_point(aes(y=y),alpha=.5)
	+ geom_line(aes(y=f),alpha=.5)
)


dx = matrix(NA,n,n)
for(i in 1:n){
	for(j in 1:n){
		dx[i,j] = -((x[i]-x[j])^2)/2
	}
}

dx_ut_flat = as.numeric(dx[upper.tri(dx)])
udx = unique(dx_ut_flat)
dx_lt_flat = as.numeric(dx[lower.tri(dx)])

i_uk_k_ut = match(dx_ut_flat,udx)
i_uk_k_lt = match(dx_lt_flat,udx)
i_k_diag = which(as.numeric(diag(1,n,n))==1)
i_k_ut = which(as.numeric(upper.tri(matrix(NA,n,n)))==1)
i_k_lt = which(as.numeric(lower.tri(matrix(NA,n,n)))==1)

data_for_stan = lst(
	n_xy = nrow(dat)
	, x = dat$x
	, y = dat$y
	, udx = udx
	, i_uk_k_ut = i_uk_k_ut
	, i_uk_k_lt = i_uk_k_lt
	, i_k_diag = i_k_diag
	, i_k_ut = i_k_ut
	, i_k_lt = i_k_lt
	, n_udx = length(udx)
	, n_ut = length(i_k_ut)
)
glimpse(data_for_stan)

#sample
sample_mod = cmdstan_model('pvgp_sample.stan',force_recompile=T)
sampled = sample_mod$sample(
	data = data_for_stan
	, chains = parallel::detectCores()/2
	, parallel_chains = parallel::detectCores()/2
	, refresh = 20
)

#check timing
p = sampled$profiles()
(
	p
	%>% bind_rows()
	%>% select(name,total_time)
	%>% group_by(name)
	%>% summarise(
		min = min(total_time)
		, max = max(total_time)
	)
)

#check errors
(
	c('err1','err2')
	%>% sampled$draws(format='draws_list')
	%>% posterior::as_draws_rvars()
)




#check diagnostics
sampled$cmdstan_diagnose()

#check summary of hyperparameters
sampled$summary(variables=c('pvn','lengthscale'))

#check pairs plot
(
	c('pvn','lengthscale')
	%>% sampled$draws()
	%>% mcmc_pairs(
		np = nuts_params(sampled)
		, off_diag_fun = 'hex'
	)
)

#extract the samples for f
(
	sampled$draws('f',format='draws_df')
	# %>% as_draws_df()
	%>% as_tibble()
	%>% pivot_longer(starts_with('f'))
	%>% separate(
		name
		, into = c('name','index')
		, convert = TRUE
	)
) -> f_samples


#join with the data and plot
(
	f_samples
	%>% left_join(dat)
	%>% ggplot(aes(x=x))
	+ geom_fan(aes(y=value),intervals=seq(.01,.99,.01))
	+ scale_fill_viridis_c(direction=-1,guide='none')
	+ geom_interval(aes(y=value),intervals=.5,linetype=3,colour='black')
	+ geom_line(data=dat,aes(y=f),colour='white',size=1)#,alpha = .8)
	+ geom_point(data=dat,aes(y=y),alpha=.5,size=2,colour='black')
	+ geom_point(data=dat,aes(y=y),alpha=.5,size=2,colour='white',shape=1)
	+ theme(
		panel.grid = element_blank()
		, panel.background = element_rect(fill='black')
	)
)
