
#ifndef EKF_H
#define EKF_H

#include "Eigen/Dense"
#include <vector>
#include <string>
#include <fstream>


class EKF {
public:
	// state vector
	Eigen::VectorXd x_;

	// state covariance matrix
	Eigen::MatrixXd P_;

	// state transistion matrix
	Eigen::MatrixXd F_;

	// process covariance matrix
	Eigen::MatrixXd Q_;

	// measurement matrix
	Eigen::MatrixXd H_;

	// measurement covariance matrix
	Eigen::MatrixXd R_;


	/**
	* Constructor
	*/
	EKF() {};

	/**
	* Destructor
	*/
	virtual ~EKF() {};


	void Init(Eigen::VectorXd &x_in, Eigen::MatrixXd &P_in, Eigen::MatrixXd &F_in,
		Eigen::MatrixXd &H_in, Eigen::MatrixXd &R_in, Eigen::MatrixXd &Q_in) {
		x_ = x_in;
		P_ = P_in;
		F_ = F_in;
		H_ = H_in;
		R_ = R_in;
		Q_ = Q_in;
	};

	void Predict() {
		x_ = F_ * x_;
		Eigen::MatrixXd Ft = F_.transpose();
		P_ = F_ * P_ * Ft + Q_;
	};

	void Update(const Eigen::VectorXd &z) {
		Eigen::VectorXd z_pred = H_ * x_;
		Eigen::VectorXd y = z - z_pred;
		Eigen::MatrixXd Ht = H_.transpose();
		Eigen::MatrixXd S = H_ * P_ * Ht + R_;
		Eigen::MatrixXd Si = S.inverse();
		Eigen::MatrixXd PHt = P_ * Ht;
		Eigen::MatrixXd K = PHt * Si;

		//new estimate
		x_ = x_ + (K * y);
		long x_size = x_.size();
		Eigen::MatrixXd I = Eigen::MatrixXd::Identity(x_size, x_size);
		P_ = (I - K * H_) * P_;
	};

	void UpdateEKF(const Eigen::VectorXd &z) {};
};

#endif /* UKF_H */
