#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 30;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 30;

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

  /**
  TODO:

  Complete the initialization. See ukf.h for other member properties.

  Hint: one or more values initialized above might be wildly off...
  */

  // middle params
  is_initialized_ = false;
  n_x_ = 5;
  n_aug_ = 7;
  lambda_ = 3 - n_aug_;

  // state params
  x_ = VectorXd(n_x_);
  P_ = MatrixXd(n_x_, n_x_);

  // state aug data params
  weights_ = VectorXd(2*n_aug_+1);
  Xsig_pred_ = MatrixXd(n_x_, 2*n_aug_+1);

}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Make sure you switch between lidar and radar
  measurements.
  */

  // Initialization
  if (!is_initialized_) {
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      // get measurements
      double l = meas_package.raw_measurements_[0];
      double alpha = meas_package.raw_measurements_[1];
      double l1 = meas_package.raw_measurements_[2];
      // init
      x_ << l*cos(alpha), l*sin(alpha), l1, 0, 0;
    } else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      double x0 = meas_package.raw_measurements_[0];
      double y0 = meas_package.raw_measurements_[1];
      x_ << x0, y0, 0, 0, 0;
    }
    // check or zeros
    if (fabs(x_(0))<0.001) {
      x_(0) = 0.001;
    }
    if (fabs(x_(1))<0.001) {
      x_(1) = 0.001;
    }
    P_ << 1, 0, 0, 0, 0,
          0, 1, 0, 0, 0,
          0, 0, 1, 0, 0,
          0, 0, 0, 1, 0,
          0, 0, 0, 0, 1;
    time_us_ = meas_package.timestamp_;
    is_initialized_ = true;

    cout<<"Initialization Finished"<<endl;
    return;
  }

  // Prediction
  double delta_t = (meas_package.timestamp_ - time_us_) / 1000000.0;
  UKF::Prediction(delta_t);

  cout<<"Prediction Finished"<<endl;

  // Update
  if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    UKF::UpdateRadar(meas_package);
  } else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
    UKF::UpdateLidar(meas_package);
  }

  time_us_ = meas_package.timestamp_;
  cout<<"Update Finished"<<endl;

}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  /**
  TODO:

  Complete this function! Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
  */

  // Augmentation of noise
  VectorXd x_aug = VectorXd(n_aug_);
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);
  //create augmented mean state
  x_aug.head(n_x_) = x_;
  x_aug(5) = 0;
  x_aug(6) = 0;
  //create augmented covariance matrix
  P_aug.fill(0.0);
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug(5,5) = std_a_*std_a_;
  P_aug(6,6) = std_yawdd_*std_yawdd_;
  //create square root matrix
  cout<<"start calu sqrt of a matrix"<<endl;
  MatrixXd A = P_aug.llt().matrixL();
  cout<<"end calu sqrt of a matrix"<<endl;
  //create augmented sigma points
  Xsig_aug.col(0) = x_aug;
  for (int i = 0; i < n_aug_; ++i) {
    Xsig_aug.col(i+1) = x_aug + sqrt(lambda_+n_aug_)*A.col(i);
    Xsig_aug.col(i+1+n_aug_) = x_aug - sqrt(lambda_+n_aug_)*A.col(i);
  }

  cout<<"P4"<<endl;
  cout<<"Iteration Number: "<<2 * n_aug_ + 1<<endl;
  // Predict Processed Sigma Points Xsig_pred
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    cout<<"Curr i Start: "<<i<<endl;
    // define some variable
    double p_x = Xsig_aug(0,i);
    double p_y = Xsig_aug(1,i);
    double v = Xsig_aug(2,i);
    double yaw = Xsig_aug(3,i);
    double yawd = Xsig_aug(4,i);
    double nu_a = Xsig_aug(5,i);
    double nu_yawdd = Xsig_aug(6,i);
    // needed variable
    double p_x2, p_y2, v2, yaw2, yawd2;
    // calculate
    if (fabs(yawd)>0.001) {
      p_x2 = p_x + v/yawd*(sin(yaw+yawd*delta_t)-sin(yaw)) + 1.0/2*delta_t*delta_t*cos(yaw)*nu_a;
      p_y2 = p_y + v/yawd*(-cos(yaw+yawd*delta_t)+cos(yaw)) + 1.0/2*delta_t*delta_t*sin(yaw)*nu_a;
      v2 = v + delta_t*nu_a;
      cout<<"----------"<<endl;
      cout<<"yaw: "<<yaw<<endl;
      cout<<"yawd: "<<yawd<<endl;
      cout<<"delta_t: "<<delta_t<<endl;
      cout<<"nu_yawdd: "<<nu_yawdd<<endl;
      cout<<"A:"<<A<<endl;
      cout<<"----------"<<endl;
      yaw2 = yaw + delta_t*yawd + 1.0/2*delta_t*delta_t*nu_yawdd;
      yawd2 = yawd + delta_t*nu_yawdd;
    }else{
      p_x2 = p_x + v*cos(yaw)*delta_t + 1.0/2*delta_t*delta_t*cos(yaw)*nu_a;
      p_y2 = p_y + v*sin(yaw)*delta_t + 1.0/2*delta_t*delta_t*sin(yaw)*nu_a;
      v2 = v + delta_t*nu_a;
      cout<<"----------"<<endl;
      cout<<"yaw: "<<yaw<<endl;
      cout<<"yawd: "<<yawd<<endl;
      cout<<"delta_t: "<<delta_t<<endl;
      cout<<"nu_yawdd: "<<nu_yawdd<<endl;
      cout<<"----------"<<endl;
      yaw2 = yaw + delta_t*yawd + 1.0/2*delta_t*delta_t*nu_yawdd;
      yawd2 = yawd + delta_t*nu_yawdd;
    }
    // update
    Xsig_pred_(0,i) = p_x2;
    Xsig_pred_(1,i) = p_y2;
    Xsig_pred_(2,i) = v2;

    cout<<"before yaw2:"<<yaw2<<endl;
    if (yaw2> M_PI) {
      yaw2 -= (long((yaw2-M_PI)/2.0/M_PI) + 1 ) *2*M_PI;
    }
    cout<<"middle yaw2: "<<yaw2<<endl;
    if (yaw2<-M_PI) {
      yaw2 += (long((-M_PI-yaw2)/2.0/M_PI)+1 ) *2*M_PI;
    }
    cout<<"after yaw2:"<<yaw2<<endl;

    Xsig_pred_(3,i) = yaw2;
    Xsig_pred_(4,i) = yawd2;
  }

  cout<<"P5"<<endl;
  // calculate mean and variance of Xsig_pred
  //set weights
  weights_(0) = lambda_ /(lambda_+n_aug_);
  for (int i=1; i<2*n_aug_+1; i++) {
    weights_(i) = 0.5/(lambda_+n_aug_);
  }
  //predict state mean
  x_.fill(0.0);
  for (int i=0; i<2*n_aug_+1; i++) {
    x_ += weights_(i)*Xsig_pred_.col(i);
  }
  //predict state covariance matrix
  P_.fill(0.0);
  for (int i=0; i<2*n_aug_+1; i++) {
    VectorXd diff = Xsig_pred_.col(i)-x_;
    //angle normalization
    if (diff(3)> M_PI) {
      diff(3) -= (int((diff(3)-M_PI)/2.0/M_PI) + 1 ) *2*M_PI;
    }
    if (diff(3)<-M_PI) {
      diff(3) += (int((-M_PI-diff(3))/2.0/M_PI)+1 ) *2*M_PI;
    }
    P_ += weights_(i)*diff*diff.transpose();
  }
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */

  MatrixXd Zsig = MatrixXd(2, 2 * n_aug_ + 1);
  VectorXd z_pred = VectorXd(2);
  MatrixXd S = MatrixXd(2, 2);
  // state sigma points to measurement sigma points
  for (int i=0; i<2*n_aug_+1; i++) {
    VectorXd curr_col = Xsig_pred_.col(i);
    double px = curr_col(0);
    double py = curr_col(1);
    double v = curr_col(2);
    double phi = curr_col(3);
    double phid = curr_col(4);

    Zsig.col(i)(0) = px;
    Zsig.col(i)(1) = py;
  }
  //calculate mean predicted measurement
  z_pred.fill(0.0);
  for (int i=0; i<2*n_aug_+1; i++) {
    z_pred += weights_(i)*Zsig.col(i);
  }
  //calculate measurement covariance matrix S
  S.fill(0.0);
  for (int i=0; i<2*n_aug_+1; i++) {
    VectorXd diff = Zsig.col(i)-z_pred;
    //angle normalization
    if (diff(1)> M_PI) {
      diff(1) -= (int((diff(1)-M_PI)/2.0/M_PI) + 1 ) *2*M_PI;
    }
    if (diff(1)<-M_PI) {
      diff(1) += (int((-M_PI-diff(1))/2.0/M_PI)+1 ) *2*M_PI;
    }
    S += weights_(i)*diff*diff.transpose();
  }
  S(0,0) += std_laspx_*std_laspx_;
  S(1,1) += std_laspy_*std_laspy_;

  // UKF update
  UKF::UKF_Update(meas_package, Zsig, z_pred, S);
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */

  MatrixXd Zsig = MatrixXd(3, 2 * n_aug_ + 1);
  VectorXd z_pred = VectorXd(3);
  MatrixXd S = MatrixXd(3, 3);
  // state sigma points to measurement sigma points
  for (int i=0; i<2*n_aug_+1; i++) {
    VectorXd curr_col = Xsig_pred_.col(i);
    double px = curr_col(0);
    double py = curr_col(1);
    double v = curr_col(2);
    double phi = curr_col(3);
    double phid = curr_col(4);

    // check or zeros
    if (fabs(px)<0.001) {
      px = 0.001;
    }
    if (fabs(py)<0.001) {
      py = 0.001;
    }

    Zsig.col(i)(0) = sqrt(px*px+py*py);
    Zsig.col(i)(1) = atan2(py, px);
    Zsig.col(i)(2) = (px*cos(phi)*v + py*sin(phi)*v) / Zsig.col(i)(0);
  }
  //calculate mean predicted measurement
  z_pred.fill(0.0);
  for (int i=0; i<2*n_aug_+1; i++) {
    z_pred += weights_(i)*Zsig.col(i);
  }
  //calculate measurement covariance matrix S
  S.fill(0.0);
  for (int i=0; i<2*n_aug_+1; i++) {
    VectorXd diff = Zsig.col(i)-z_pred;
    //angle normalization
    if (diff(1)> M_PI) {
      diff(1) -= (int((diff(1)-M_PI)/2.0/M_PI) + 1 ) *2*M_PI;
    }
    if (diff(1)<-M_PI) {
      diff(1) += (int((-M_PI-diff(1))/2.0/M_PI)+1 ) *2*M_PI;
    }
    S += weights_(i)*diff*diff.transpose();
  }
  S(0,0) += std_radr_*std_radr_;
  S(1,1) += std_radphi_*std_radphi_;
  S(2,2) += std_radrd_*std_radrd_;

  // UKF update
  UKF::UKF_Update(meas_package, Zsig, z_pred, S);
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 * @param {Zsig} sigma points in measurement space
 * @param {z_pred} mean predicted measurement
 * @param {S} predicted measurement covariance
 */
void UKF::UKF_Update(MeasurementPackage meas_package, MatrixXd Zsig, VectorXd z_pred, MatrixXd S) {

  // get current measurement z
  VectorXd z = meas_package.raw_measurements_;

  //calculate cross correlation matrix
  MatrixXd T = MatrixXd(n_x_, z.size());
  T.fill(0.0);
  for (int i=0; i<2*n_aug_+1; i++){
    VectorXd x_diff = Xsig_pred_.col(i)-x_;
    VectorXd z_diff = Zsig.col(i)-z_pred;
    // angle normalization
    if (z_diff(1)> M_PI) {
      z_diff(1) -= (int((z_diff(1)-M_PI)/2.0/M_PI) + 1 ) *2*M_PI;
    }
    if (z_diff(1)<-M_PI) {
      z_diff(1) += (int((-M_PI-z_diff(1))/2.0/M_PI)+1 ) *2*M_PI;
    }
    if (x_diff(3)> M_PI) {
      x_diff(3) -= (int((x_diff(3)-M_PI)/2.0/M_PI) + 1 ) *2*M_PI;
    }
    if (x_diff(3)<-M_PI) {
      x_diff(3) += (int((-M_PI-x_diff(3))/2.0/M_PI)+1 ) *2*M_PI;
    }
    T += weights_(i) * x_diff * z_diff.transpose();
  }

  //calculate Kalman gain K;
  MatrixXd K = T*S.inverse();

  //update state mean and covariance matrix
  VectorXd z_diff = z - z_pred;

  //angle normalization
  if (z_diff(1)> M_PI) {
    z_diff(1) -= (int((z_diff(1)-M_PI)/2.0/M_PI) + 1 ) *2*M_PI;
  }
  if (z_diff(1)<-M_PI) {
    z_diff(1) += (int((-M_PI-z_diff(1))/2.0/M_PI)+1 ) *2*M_PI;
  }
  x_ += K*(z_diff);
  P_ -= K*S*K.transpose();
}