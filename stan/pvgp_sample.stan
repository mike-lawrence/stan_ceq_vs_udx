functions{
	matrix gp_exp_quad_cov_from_udx(
		real eta
		, real rho
		, real jitter
		, int n_x
		, int square_n_x
		, int n_udx
		, vector udx
		, int[] i_uk_k_ut
		, int[] i_uk_k_lt
		, int[] i_k_diag
		, int[] i_k_ut
		, int[] i_k_lt
	){
		real sq_eta = square( eta ) ;
		vector[n_udx] uk = sq_eta * exp( udx / square(rho) ) ;
		vector[square_n_x] k_flat ;
		k_flat[i_k_diag] = rep_vector(sq_eta + jitter,n_x) ;
		k_flat[i_k_ut] = uk[i_uk_k_ut] ;
		k_flat[i_k_lt] = uk[i_uk_k_lt] ;
		return to_matrix(k_flat,n_x,n_x) ;
	}
	matrix unit_gp_exp_quad_cov_from_udx(
		real rho
		, real jitter
		, int n_x
		, int square_n_x
		, int n_udx
		, vector udx
		, int[] i_uk_k_ut
		, int[] i_uk_k_lt
		, int[] i_k_diag
		, int[] i_k_ut
		, int[] i_k_lt
	){
		vector[n_udx] uk = exp( udx / square(rho) ) ;
		vector[square_n_x] k_flat ;
		k_flat[i_k_diag] = rep_vector(1 + jitter,n_x) ;
		k_flat[i_k_ut] = uk[i_uk_k_ut] ;
		k_flat[i_k_lt] = uk[i_uk_k_lt] ;
		return to_matrix(k_flat,n_x,n_x) ;
	}

}
data{

	// n_xy: number of observations
	int<lower=1> n_xy ;

	// y: outcome variable
	vector[n_xy] y ;

	// x: continuous variable whose influence on y we're modelling as a GP
	real x[n_xy] ;

	//n_udx: number of unique differences in x
	int n_udx ;

	// udx: vector of unique differences in x (squared & divided by -2)
	vector[n_udx] udx ;

	// i_k_diag: index array indicating elements in flattened covariance matrix that are on the diagonal when unflattened
	array[n_xy] int i_k_diag ;
	// n_ut: number of elements in covariance upper triangle
	int n_ut ;
	// i_k_ut: index array indicating elements in flattened covariance matrix that are in the upper triangle when unflattened
	array[n_ut] int i_k_ut ;
	// i_k_lt: index array indicating elements in flattened covariance matrix that are in the lower triangle when unflattened
	array[n_ut] int i_k_lt ;
	// i_uk_k_ut: index array mapping between unique entries in k to the upper-triangle of k
	array[n_ut] int i_uk_k_ut ;
	// i_uk_k_lt: index array mapping between unique entries in k to the lower-triangle of k
	array[n_ut] int i_uk_k_lt ;


}
transformed data{
	int square_n_xy = n_xy*n_xy ;

	// pre-compute some quantities:
	//   - to make the priors somewhat-automated/standardized, we'll "scale" y by
	//     subtracting the observed mean then dividing by the observed SD, saving
	//     these quantities for later unscaling the model output. I add a "_"
	//     suffix to all variables that are on this transformed scale to
	//     differentiate from their unscaled counterparts.
	//   - min_dist & dist_range will help express a prior for the GP "lengthscale"
	//     parameter in a manner that avoids areas of the parameter space on which
	//     the data cannot inform.
	real obs_y_sd = sd(y) ;
	real obs_y_mean = mean(y) ;
	vector[n_xy] y_ = (y - obs_y_mean)/obs_y_sd ;
	real max_dist = max(x)-min(x);
	real min_dist = max_dist ; //initializing at max, but will get smaller in loop
	for(i_n_xy in 1:(n_xy-1)){
		for(j_n_xy in (i_n_xy+1):n_xy){
			real dist = fabs(x[i_n_xy]-x[j_n_xy]) ;
			if(dist<min_dist){
				min_dist = dist ;
			}
		}
	}
	real dist_range = max_dist - min_dist ;
}
parameters{
	// logit_pvn: logit proportion of variance attributable to noise
	real logit_pvn ;
	// logit_p_lengthscale: logit-transformed GP lengthscale, expressed as as a
	//  proportion of the data-imformable-range
	real logit_p_lengthscale ;
	// helper variable for the GP that will be given a std_normal prior
	vector[n_xy] f_stdnormal ;
}
transformed parameters{
	real pvn ;
	real lengthscale ;
	profile("tp"){
		// pvn: reversing the logit transform
		pvn = inv_logit(logit_pvn) ;
		// lengthscale: reversing logit & scaling/shifting to yield desired bounds
		lengthscale = inv_logit(logit_p_lengthscale)*dist_range + min_dist ;
	}
	// f_: (scaled) latent GP
	vector[n_xy] f_ ;
	real delta;
	{
		// cov_mat: covariance matrix for the GP, using the standard "exponentiated quadratic" kernel
		matrix[n_xy,n_xy] k_eqc ;
		matrix[n_xy,n_xy] k_udx ;
		matrix[n_xy,n_xy] k_uudx ;
		profile("eqc"){
			k_eqc = add_diag(
				gp_exp_quad_cov(x,1.0,lengthscale)
				, 1e-5
			) ;
		}
		profile("udx"){
			k_udx = unit_gp_exp_quad_cov_from_udx(
				  lengthscale // real rho
				, 1e-5 // real jitter
				, n_xy // int n_x
				, square_n_xy // int square_n_x
				, n_udx // int n_udx
				, udx // vector udx
				, i_uk_k_ut // int i_uk_k_ut[]
				, i_uk_k_lt // int i_uk_k_lt[]
				, i_k_diag // int i_k_diag[]
				, i_k_ut // int i_k_ut[]
				, i_k_lt // int i_k_lt[]
			) ;
		}
		delta = max(square(k_eqc-k_udx));
		profile("uudx"){
			k_uudx = unit_gp_exp_quad_cov_from_udx(
				lengthscale // real rho
				, 1e-5 // real jitter
				, n_xy // int n_x
				, square_n_xy // int square_n_x
				, n_udx // int n_udx
				, udx // vector udx
				, i_uk_k_ut // int i_uk_k_ut[]
				, i_uk_k_lt // int i_uk_k_lt[]
				, i_k_diag // int i_k_diag[]
				, i_k_ut // int i_k_ut[]
				, i_k_lt // int i_k_lt[]
			) ;
		}
		// compute f_ as the product of the cholesky-decomposed correlations and
		//   the standard-normal variate, yielding a set of values that vary smoothly
		//   as a function of x. Final multiplication by sqrt(1-pvn) pairs with
		//   y_~normal(f_,sqrt(pvn)) below to encode the proportion-of-variance
		//   parameterization.
		profile("f_"){
			f_ = (
				cholesky_decompose(k_eqc)
				* f_stdnormal
				* sqrt(1-pvn)
			) ;
		}
	}
}
model{
	profile("model"){
		// Prior on logit_p-lengthscale peaked at 0 implies credibility peaked at
		//   the middle of the range on which the data can inform. A SD of 1 implies
		//   tapering of credibility towards the bounds of the range in a manner that
		//   distributes credibility across a reasonably broad range of GP "wiggliness".
		//   Could have used std_normal() but for one parameter there's virtually no
		//   speedup and leaving as normal(0,1) highlights that the SD (or even the
		//   mean) could be tweaked to more faithfully encode domain expertise as
		//   necessary. Changing the mean will shift the peak credibility to higher
		//   or lower lengthscales, and changing the SD will concentrate/spread
		//   credibility. Note that normal(0,1.5) yields appox uniform credibility
		//   and normal(0,2) will have a u-shape such that the middle of the range
		//   has lower credibility than either extreme.
		logit_p_lengthscale ~ normal(0,1) ;
		// Prior on pvn; ditto to the last comment :)
		logit_pvn ~ normal(0,1) ;
		// f_stdnormal implicitly encodes structure for the latent GP and *must* be
		//   std_normal()
		f_stdnormal ~ std_normal() ;
		// "Likelihood" whereby y_ is centered on f_ with an SD of sqrt(pvn)
		y_ ~ normal(f_,sqrt(pvn)) ;
	}
}
generated quantities{
	// f: (unscaled) latent GP
	vector[n_xy] f = f_*obs_y_sd + obs_y_mean ;
}
