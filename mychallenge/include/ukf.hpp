#ifndef UKF_H
#define UKF_H

#include "Eigen/Dense"
#include <vector>
#include <string>
#include <fstream>


class UKF {
public:

	///* initially set to false, set to true in first call of ProcessMeasurement
	bool is_initialized_;

	///* if this is false, laser measurements will be ignored (except for init)
	bool use_laser_;

	///* if this is false, radar measurements will be ignored (except for init)
	bool use_radar_;

	///* state vector: [pos1 pos2 vel_abs yaw_angle yaw_rate] in SI units and rad
	Eigen::VectorXd x_;

	///* state covariance matrix
	Eigen::MatrixXd P_;

	///* predicted sigma points matrix
	Eigen::MatrixXd Xsig_pred_;

	///* time when the state is true, in us
	long long time_us_;

	///* Process noise standard deviation longitudinal acceleration in m/s^2
	double std_a_;

	///* Process noise standard deviation yaw acceleration in rad/s^2
	double std_yawdd_;

	///* Laser measurement noise standard deviation position1 in m
	double std_laspx_;

	///* Laser measurement noise standard deviation position2 in m
	double std_laspy_;

	///* Radar measurement noise standard deviation radius in m
	double std_radr_;

	///* Radar measurement noise standard deviation angle in rad
	double std_radphi_;

	///* Radar measurement noise standard deviation radius change in m/s
	double std_radrd_;

	///* Weights of sigma points
	Eigen::VectorXd weights_;

	///* State dimension
	int n_x_;

	///* Augmented state dimension
	int n_aug_;

	///* Sigma point spreading parameter
	double lambda_;


	/**
	* Constructor
	*/
	UKF() {
		// if this is false, laser measurements will be ignored (except during init)
		use_laser_ = true;

		// if this is false, radar measurements will be ignored (except during init)
		use_radar_ = true;

		// initial state vector
		x_ = Eigen::VectorXd(5);

		// initial covariance matrix
		P_ = Eigen::MatrixXd(5, 5);

		// Process noise standard deviation longitudinal acceleration in m/s^2
		std_a_ = 2;

		// Process noise standard deviation yaw acceleration in rad/s^2
		std_yawdd_ = 0.5;

		// Laser measurement noise standard deviation position1 in m
		std_laspx_ = 0.15;

		// Laser measurement noise standard deviation position2 in m
		std_laspy_ = 0.15;

		// Radar measurement noise standard deviation radius in m
		std_radr_ = 0.3;

		// Radar measurement noise standard deviation angle in rad
		std_radphi_ = 0.03;

		// Radar measurement noise standard deviation radius change in m/s
		std_radrd_ = 0.3;

		n_x_ = 5;
		n_aug_ = 7;

		Xsig_pred_ = Eigen::MatrixXd(n_x_, 2 * n_aug_ + 1);

		weights_ = Eigen::VectorXd(2 * n_aug_ + 1);

		lambda_ = 3 - n_aug_;


		// set weights
		double weight_0 = lambda_ / (lambda_ + n_aug_);
		weights_(0) = weight_0;
		for (int i = 1; i < 2 * n_aug_ + 1; i++) {  //2n+1 weights
			double weight = 0.5 / (n_aug_ + lambda_);
			weights_(i) = weight;
		}

	};

	/**
	* Destructor
	*/
	virtual ~UKF() {};

	/**
	* ProcessMeasurement
	* @param meas_package The latest measurement data of either radar or laser
	*/
	void Update(Eigen::VectorXd measurements) {
		//set state dimension
		int n_x = n_x_;

		//set augmented dimension
		int n_aug = n_aug_;

		//set measurement dimension, radar can measure r, phi, and r_dot
		int n_z = 2;

		//create matrix for sigma points in measurement space
		Eigen::MatrixXd Zsig = Eigen::MatrixXd(n_z, 2 * n_aug + 1);

		//transform sigma points into measurement space
		for (int i = 0; i < 2 * n_aug + 1; i++) {  //2n+1 simga points

												   // extract values for better readibility
			double p_x = Xsig_pred_(0, i);
			double p_y = Xsig_pred_(1, i);
			double v = Xsig_pred_(2, i);
			double yaw = Xsig_pred_(3, i);

			double v1 = cos(yaw)*v;
			double v2 = sin(yaw)*v;

			// measurement model
			Zsig(0, i) = p_x;                        //px
			Zsig(1, i) = p_y;                        //py
		}

		//mean predicted measurement
		Eigen::VectorXd z_pred = Eigen::VectorXd(n_z);
		z_pred.fill(0.0);
		for (int i = 0; i < 2 * n_aug + 1; i++) {
			z_pred = z_pred + weights_(i) * Zsig.col(i);
		}

		//measurement covariance matrix S
		Eigen::MatrixXd S = Eigen::MatrixXd(n_z, n_z);
		S.fill(0.0);
		for (int i = 0; i < 2 * n_aug + 1; i++) {  //2n+1 simga points
												   //residual
			Eigen::VectorXd z_diff = Zsig.col(i) - z_pred;

			//      //angle normalization
			//      while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
			//      while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

			S = S + weights_(i) * z_diff * z_diff.transpose();
		}

		//add measurement noise covariance matrix
		Eigen::MatrixXd R = Eigen::MatrixXd(n_z, n_z);
		R << std_laspx_*std_laspx_, 0,
			0, std_laspy_*std_laspy_;
		S = S + R;

		Eigen::VectorXd z = measurements;

		//create matrix for cross correlation Tc
		Eigen::MatrixXd Tc = Eigen::MatrixXd(n_x, n_z);

		//calculate cross correlation matrix
		Tc.fill(0.0);
		for (int i = 0; i < 2 * n_aug + 1; i++) {  //2n+1 simga points

												   //residual
			Eigen::VectorXd z_diff = Zsig.col(i) - z_pred;
			//      //angle normalization
			//      while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
			//      while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

			// state difference
			Eigen::VectorXd x_diff = Xsig_pred_.col(i) - x_;
			//angle normalization
			//  while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
			//  while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;
			if (x_diff(3) > M_PI)
				x_diff(3) = x_diff(3) - 2 * M_PI - ((int)((x_diff(3) - M_PI) / 2 / M_PI))*2.*M_PI;
			if (x_diff(3) < -M_PI)
				x_diff(3) = x_diff(3) + 2 * M_PI - ((int)((x_diff(3) + M_PI) / 2 / M_PI))*2.*M_PI;

			Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
		}

		//Kalman gain K;
		Eigen::MatrixXd K = Tc * S.inverse();

		//    //residual
		Eigen::VectorXd z_diff = z - z_pred;
		//NIScost = z_diff.transpose() * S.inverse() * z_diff;

		//    //angle normalization
		//    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
		//    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

		//update state mean and covariance matrix
		x_ = x_ + K * z_diff;
		// while (x_(3)> M_PI) x_(3)-=2.*M_PI;
		// while (x_(3)<-M_PI) x_(3)+=2.*M_PI;
		// std::cout<<"yaw1:"<<x_(3);
		//     if (x_(3)>M_PI)
		//       x_(3) = x_(3) - 2.*M_PI - ((int)((x_(3)-M_PI)/2./M_PI))*2.*M_PI;
		//     if (x_(3)<-M_PI)
		//       x_(3) = x_(3) + 2.*M_PI - ((int)((x_(3)+M_PI)/2./M_PI))*2.*M_PI;
		if (x_(4) > 10 || x_(4) < -10)
			x_(4) = 0;
		// std::cout<<"yaw2:"<<x_(3);
		P_ = P_ - K*S*K.transpose();

	};


	/**
	* ProcessMeasurement
	* @param meas_package The latest measurement data of either radar or laser
	*/
	void UpdateRadar(Eigen::VectorXd measurements) {
		//set state dimension
		int n_x = n_x_;

		//set augmented dimension
		int n_aug = n_aug_;

		//set measurement dimension, radar can measure r, phi, and r_dot
		int n_z = 3;

		//create matrix for sigma points in measurement space
		Eigen::MatrixXd Zsig = Eigen::MatrixXd(n_z, 2 * n_aug + 1);

		//transform sigma points into measurement space
		for (int i = 0; i < 2 * n_aug + 1; i++) {  //2n+1 simga points

												   // extract values for better readibility
			double p_x = Xsig_pred_(0, i);
			double p_y = Xsig_pred_(1, i);
			double v = Xsig_pred_(2, i);
			double yaw = Xsig_pred_(3, i);

			double v1 = cos(yaw)*v;
			double v2 = sin(yaw)*v;

			// measurement model
			Zsig(0, i) = sqrt(p_x*p_x + p_y*p_y);                        //r
			Zsig(1, i) = atan2(p_y, p_x);                                 //phi
			Zsig(2, i) = (p_x*v1 + p_y*v2) / sqrt(p_x*p_x + p_y*p_y);   //r_dot
		}
		//mean predicted measurement
		Eigen::VectorXd z_pred = Eigen::VectorXd(n_z);
		z_pred.fill(0.0);
		for (int i = 0; i < 2 * n_aug + 1; i++) {
			z_pred = z_pred + weights_(i) * Zsig.col(i);
		}
		;
		//measurement covariance matrix S
		Eigen::MatrixXd S = Eigen::MatrixXd(n_z, n_z);
		S.fill(0.0);
		for (int i = 0; i < 2 * n_aug + 1; i++) {  //2n+1 simga points
												   //residual
			Eigen::VectorXd z_diff = Zsig.col(i) - z_pred;

			//      //angle normalization
			//      while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
			//      while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;
			if (z_diff(1) > M_PI)
				z_diff(1) = z_diff(1) - 2 * M_PI - ((int)((z_diff(1) - M_PI) / 2 / M_PI))*2.*M_PI;
			if (z_diff(1) < -M_PI)
				z_diff(1) = z_diff(1) + 2 * M_PI - ((int)((z_diff(1) + M_PI) / 2 / M_PI))*2.*M_PI;

			S = S + weights_(i) * z_diff * z_diff.transpose();
		}

		//add measurement noise covariance matrix
		Eigen::MatrixXd R = Eigen::MatrixXd(n_z, n_z);
		R << std_radr_*std_radr_, 0, 0,
			0, std_radphi_*std_radphi_, 0,
			0, 0, std_radrd_*std_radrd_;
		S = S + R;
		Eigen::VectorXd z = measurements;

		//create matrix for cross correlation Tc
		Eigen::MatrixXd Tc = Eigen::MatrixXd(n_x, n_z);

		//calculate cross correlation matrix
		Tc.fill(0.0);
		for (int i = 0; i < 2 * n_aug + 1; i++) {  //2n+1 simga points

												   //residual
			Eigen::VectorXd z_diff = Zsig.col(i) - z_pred;
			//      //angle normalization
			//      while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
			//      while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;
			if (z_diff(1) > M_PI)
				z_diff(1) = z_diff(1) - 2 * M_PI - ((int)((z_diff(1) - M_PI) / 2 / M_PI))*2.*M_PI;
			if (z_diff(1) < -M_PI)
				z_diff(1) = z_diff(1) + 2 * M_PI - ((int)((z_diff(1) + M_PI) / 2 / M_PI))*2.*M_PI;

			// state difference
			Eigen::VectorXd x_diff = Xsig_pred_.col(i) - x_;
			//angle normalization
			//  while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
			//  while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;
			if (x_diff(3) > M_PI)
				x_diff(3) = x_diff(3) - 2 * M_PI - ((int)((x_diff(3) - M_PI) / 2 / M_PI))*2.*M_PI;
			if (x_diff(3) < -M_PI)
				x_diff(3) = x_diff(3) + 2 * M_PI - ((int)((x_diff(3) + M_PI) / 2 / M_PI))*2.*M_PI;

			Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
		}

		//Kalman gain K;
		Eigen::MatrixXd K = Tc * S.inverse();

		//    //residual
		Eigen::VectorXd z_diff = z - z_pred;
		//NIScost = z_diff.transpose() * S.inverse() * z_diff;

		//    //angle normalization
		//    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
		//    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;
		if (z_diff(1) > M_PI)
			z_diff(1) = z_diff(1) - 2 * M_PI - ((int)((z_diff(1) - M_PI) / 2 / M_PI))*2.*M_PI;
		if (z_diff(1) < -M_PI)
			z_diff(1) = z_diff(1) + 2 * M_PI - ((int)((z_diff(1) + M_PI) / 2 / M_PI))*2.*M_PI;

		//update state mean and covariance matrix
		x_ = x_ + K * z_diff;
		// while (x_(3)> M_PI) x_(3)-=2.*M_PI;
		// while (x_(3)<-M_PI) x_(3)+=2.*M_PI;
		// std::cout<<"yaw1:"<<x_(3);
		//     if (x_(3)>M_PI)
		//       x_(3) = x_(3) - 2.*M_PI - ((int)((x_(3)-M_PI)/2./M_PI))*2.*M_PI;
		//     if (x_(3)<-M_PI)
		//       x_(3) = x_(3) + 2.*M_PI - ((int)((x_(3)+M_PI)/2./M_PI))*2.*M_PI;
		if (x_(4) > 10 || x_(4) < -10)
			x_(4) = 0;
		P_ = P_ - K*S*K.transpose();
		// std::cout<<"yaw2:"<<x_(3);

	};

	/**
	* Prediction Predicts sigma points, the state, and the state covariance
	* matrix
	* @param delta_t Time between k and k+1 in s
	*/
	void Prediction(double delta_t) {
		int n_x = n_x_;
		//define spreading parameter
		double lambda = 3 - n_x;

		//create sigma point matrix
		Eigen::MatrixXd Xsig = Eigen::MatrixXd(n_x, 2 * n_x + 1);

		//calculate square root of P
		Eigen::MatrixXd A = P_.llt().matrixL();

		//set first column of sigma point matrix
		Xsig.col(0) = x_;

		//set remaining sigma points
		for (int i = 0; i < n_x; i++)
		{
			Xsig.col(i + 1) = x_ + sqrt(lambda + n_x) * A.col(i);
			Xsig.col(i + 1 + n_x) = x_ - sqrt(lambda + n_x) * A.col(i);
		}

		int n_aug = n_aug_;

		//define spreading parameter
		lambda = 3 - n_aug;

		//create augmented mean vector
		Eigen::VectorXd x_aug = Eigen::VectorXd(7);

		//create augmented state covariance
		Eigen::MatrixXd P_aug = Eigen::MatrixXd(7, 7);

		//create sigma point matrix
		Eigen::MatrixXd Xsig_aug = Eigen::MatrixXd(n_aug, 2 * n_aug + 1);

		//create augmented mean state
		x_aug.head(5) = x_;
		x_aug(5) = 0;
		x_aug(6) = 0;

		//create augmented covariance matrix
		P_aug.fill(0.0);
		P_aug.topLeftCorner(5, 5) = P_;
		P_aug(5, 5) = std_a_*std_a_;
		P_aug(6, 6) = std_yawdd_*std_yawdd_;

		//create square root matrix
		Eigen::MatrixXd L = P_aug.llt().matrixL();

		//create augmented sigma points
		Xsig_aug.col(0) = x_aug;
		for (int i = 0; i < n_aug; i++)
		{
			Xsig_aug.col(i + 1) = x_aug + sqrt(lambda + n_aug) * L.col(i);
			Xsig_aug.col(i + 1 + n_aug) = x_aug - sqrt(lambda + n_aug) * L.col(i);
		}

		//predict sigma points
		for (int i = 0; i < 2 * n_aug + 1; i++)
		{
			//extract values for better readability
			double p_x = Xsig_aug(0, i);
			double p_y = Xsig_aug(1, i);
			double v = Xsig_aug(2, i);
			double yaw = Xsig_aug(3, i);
			double yawd = Xsig_aug(4, i);
			double nu_a = Xsig_aug(5, i);
			double nu_yawdd = Xsig_aug(6, i);

			//predicted state values
			double px_p, py_p;

			//avoid division by zero
			if (fabs(yawd) > 0.001) {
				px_p = p_x + v / yawd * (sin(yaw + yawd*delta_t) - sin(yaw));
				py_p = p_y + v / yawd * (cos(yaw) - cos(yaw + yawd*delta_t));
			}
			else {
				px_p = p_x + v*delta_t*cos(yaw);
				py_p = p_y + v*delta_t*sin(yaw);
			}

			double v_p = v;
			double yaw_p = yaw + yawd*delta_t;
			double yawd_p = yawd;

			//add noise
			px_p = px_p + 0.5*nu_a*delta_t*delta_t * cos(yaw);
			py_p = py_p + 0.5*nu_a*delta_t*delta_t * sin(yaw);
			v_p = v_p + nu_a*delta_t;

			yaw_p = yaw_p + 0.5*nu_yawdd*delta_t*delta_t;
			yawd_p = yawd_p + nu_yawdd*delta_t;

			//write predicted sigma point into right column
			Xsig_pred_(0, i) = px_p;
			Xsig_pred_(1, i) = py_p;
			Xsig_pred_(2, i) = v_p;
			Xsig_pred_(3, i) = yaw_p;
			Xsig_pred_(4, i) = yawd_p;
		}


		//predicted state mean
		x_.fill(0.0);
		for (int i = 0; i < 2 * n_aug + 1; i++) {  //iterate over sigma points
			x_ = x_ + weights_(i) * Xsig_pred_.col(i);
		}

		//predicted state covariance matrix
		P_.fill(0.0);
		for (int i = 0; i < 2 * n_aug + 1; i++) {  //iterate over sigma points

												   // state difference
			Eigen::VectorXd x_diff = Xsig_pred_.col(i) - x_;
			//angle normalization
			//while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
			//while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;
			if (x_diff(3) > M_PI)
				x_diff(3) = x_diff(3) - 2.*M_PI - ((int)((x_diff(3) - M_PI) / 2. / M_PI))*2.*M_PI;
			if (x_diff(3) < -M_PI)
				x_diff(3) = x_diff(3) + 2.*M_PI - ((int)((x_diff(3) + M_PI) / 2. / M_PI))*2.*M_PI;

			P_ = P_ + weights_(i) * x_diff * x_diff.transpose();
		}
	};

	/**
	* Prediction Predicts sigma points, and the state
	* matrix
	* @param delta_t Time between k and k+1 in s
	*/
	void Prediction(double delta_t, float msensor, float mx, float my, float& px, float& py, float& po) {
		int n_x = n_x_;
		//define spreading parameter
		double lambda = 3 - n_x;

		//create sigma point matrix
		Eigen::MatrixXd Xsig = Eigen::MatrixXd(n_x, 2 * n_x + 1);

		//calculate square root of P
		Eigen::MatrixXd A = P_.llt().matrixL();

		Eigen::VectorXd temp_x = Eigen::VectorXd(n_x);
		temp_x = x_;
		if (msensor == 1) {
			temp_x(0) = mx;
			temp_x(1) = my;
		}

		//set first column of sigma point matrix
		Xsig.col(0) = temp_x;

		//set remaining sigma points
		for (int i = 0; i < n_x; i++)
		{
			Xsig.col(i + 1) = temp_x + sqrt(lambda + n_x) * A.col(i);
			Xsig.col(i + 1 + n_x) = temp_x - sqrt(lambda + n_x) * A.col(i);
		}

		int n_aug = n_aug_;

		//define spreading parameter
		lambda = 3 - n_aug;

		//create augmented mean vector
		Eigen::VectorXd x_aug = Eigen::VectorXd(7);

		//create augmented state covariance
		Eigen::MatrixXd P_aug = Eigen::MatrixXd(7, 7);

		//create sigma point matrix
		Eigen::MatrixXd Xsig_aug = Eigen::MatrixXd(n_aug, 2 * n_aug + 1);

		//create augmented mean state
		x_aug.head(5) = temp_x;
		x_aug(5) = 0;
		x_aug(6) = 0;

		//create augmented covariance matrix
		P_aug.fill(0.0);
		P_aug.topLeftCorner(5, 5) = P_;
		P_aug(5, 5) = std_a_*std_a_;
		P_aug(6, 6) = std_yawdd_*std_yawdd_;

		//create square root matrix
		Eigen::MatrixXd L = P_aug.llt().matrixL();

		//create augmented sigma points
		Xsig_aug.col(0) = x_aug;
		for (int i = 0; i < n_aug; i++)
		{
			Xsig_aug.col(i + 1) = x_aug + sqrt(lambda + n_aug) * L.col(i);
			Xsig_aug.col(i + 1 + n_aug) = x_aug - sqrt(lambda + n_aug) * L.col(i);
		}

		Eigen::MatrixXd Xsig_pred_t = Eigen::MatrixXd(n_x, 2 * n_aug + 1);
		//predict sigma points
		for (int i = 0; i < 2 * n_aug + 1; i++)
		{
			//extract values for better readability
			double p_x = Xsig_aug(0, i);
			double p_y = Xsig_aug(1, i);
			double v = Xsig_aug(2, i);
			double yaw = Xsig_aug(3, i);
			double yawd = Xsig_aug(4, i);
			double nu_a = Xsig_aug(5, i);
			double nu_yawdd = Xsig_aug(6, i);

			//predicted state values
			double px_p, py_p;

			//avoid division by zero
			if (fabs(yawd) > 0.001) {
				px_p = p_x + v / yawd * (sin(yaw + yawd*delta_t) - sin(yaw));
				py_p = p_y + v / yawd * (cos(yaw) - cos(yaw + yawd*delta_t));
			}
			else {
				px_p = p_x + v*delta_t*cos(yaw);
				py_p = p_y + v*delta_t*sin(yaw);
			}

			double v_p = v;
			double yaw_p = yaw + yawd*delta_t;
			double yawd_p = yawd;

			//add noise
			px_p = px_p + 0.5*nu_a*delta_t*delta_t * cos(yaw);
			py_p = py_p + 0.5*nu_a*delta_t*delta_t * sin(yaw);
			v_p = v_p + nu_a*delta_t;

			yaw_p = yaw_p + 0.5*nu_yawdd*delta_t*delta_t;
			yawd_p = yawd_p + nu_yawdd*delta_t;

			//write predicted sigma point into right column
			Xsig_pred_t(0, i) = px_p;
			Xsig_pred_t(1, i) = py_p;
			Xsig_pred_t(2, i) = v_p;
			Xsig_pred_t(3, i) = yaw_p;
			Xsig_pred_t(4, i) = yawd_p;
		}



		//predicted state mean
		temp_x.fill(0.0);
		for (int i = 0; i < 2 * n_aug + 1; i++) {  //iterate over sigma points
			temp_x = temp_x + weights_(i) * Xsig_pred_t.col(i);
		}
		px = temp_x(0);
		py = temp_x(1);
		po = temp_x(3);
		if (po > M_PI)
			po = po - 2.*M_PI - ((int)((po - M_PI) / 2. / M_PI))*2.*M_PI;
		if (po < -M_PI)
			po = po + 2.*M_PI - ((int)((po + M_PI) / 2. / M_PI))*2.*M_PI;
		if (po > 100 || po < -100)
			po = 0;
	};

};

#endif /* UKF_H */
