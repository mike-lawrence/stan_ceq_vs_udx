data{
	// n: number of observations
	int<lower=1> n ;
	// x: continuous variable whose influence on y we're modelling as a GP
	real x[n] ;
	real pvn_ ;
	real lengthscale_ ;
	int<lower=0,upper=1> sample_pvn ;
	int<lower=0,upper=1> sample_lengthscale ;
}
transformed data{
	real max_dist = max(x)-min(x);
	real min_dist = max_dist ;
	for(i_n in 1:(n-1)){
		for(j_n in (i_n+1):n){
			real dist = fabs(x[i_n]-x[j_n]) ;
			if(dist<min_dist){
				min_dist = dist ;
			}
		}
	}
	real dist_range = max_dist - min_dist ;
}
parameters{
	real dummy ;
}
generated quantities{
	vector[n] y_ ;
	vector[n] f_ ;
	vector[n] y ;
	vector[n] f ;
	real obs_y_mean = std_normal_rng() ;
	real obs_y_sd = lognormal_rng(0,1) ;
	real lengthscale ;
	real logit_p_lengthscale ;
	if(sample_lengthscale==1){
		logit_p_lengthscale = std_normal_rng() ;
		lengthscale = inv_logit(logit_p_lengthscale)*dist_range + min_dist ;
	}else{
		lengthscale = lengthscale_ ;
		// logit_p_lengthscale = logit((lengthscale-min_dist)/dist_range) ;
	}
	real logit_pvn ;
	real pvn ;
	if(sample_pvn==1){
		logit_pvn = std_normal_rng() ;
		pvn = inv_logit(logit_pvn);
	}else{
		pvn = pvn_ ;
		logit_pvn = logit(pvn) ;
	}
	{
		vector[n] f_stdnormal ;
		vector[n] noise_stdnormal ;
		matrix[n,n] cov_mat = cov_exp_quad(x,1.0,lengthscale) ;
		for(i_n in 1:n){
			cov_mat[i_n,i_n] += 1e-5 ;
			f_stdnormal[i_n] = std_normal_rng() ;
			noise_stdnormal[i_n] = std_normal_rng() ;
		}
		f_ = (
			cholesky_decompose(cov_mat)
			* f_stdnormal
		) ;
		f_ = f_* sqrt(1-pvn) ;
		noise_stdnormal = noise_stdnormal*sqrt(pvn) ;
		y_ = f_ + noise_stdnormal ;
		y = y_*obs_y_sd + obs_y_mean ;
		f = f_*obs_y_sd + obs_y_mean ;
	}
}
