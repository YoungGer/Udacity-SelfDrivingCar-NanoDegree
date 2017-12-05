#include "PID.h"

using namespace std;

/*
* TODO: Complete the PID class.
*/
PID::PID() {

}

PID::~PID() {

}

void PID::Init(double Kp, double Ki, double Kd) {
    this->Kp = Kp;
    this->Ki = Ki;
    this->Kd = Kd;

    p_error = 0.0;
    i_error = 0.0;
    d_error = 0.0;
    
    step = 0;
}


void PID::UpdateError(double cte) {
    if (step==0){
        cte_prev = cte;
    }
    // pid update
    p_error = cte;
    i_error += cte;
    d_error = cte - cte_prev;
    // other update
    cte_prev = cte;
    step = step + 1;

}

double PID::TotalError() {
    return Kp * p_error + Ki * i_error + Kd * d_error;
}

