# Nonlinear Model Predictive Control (NMPC) with actuator latency

Self-Driving Car Engineer Nanodegree Program

---

## Project Description

The purpose of this project is to implement Nonlinear Model Predictive Control (NMPC) algorithm so that the car can run according to a fixed trajectory. 

The data which the simulator provides is in a global coordinate system which can been seen below:
          
- the position of the car (x and y)
- the speed of the car
- the heading direction of the car
- the coordinates of waypoints along a reference trajectory 

With all the provided data, we need to calculate two things:

-  steering angle
- acceleration (throttle/brake combined)

One thing which has to be aware of is that there is 100ms latency between actuator commands and actuator execution. The latency mimics reality and adds some difficulties to the project.

## Project Route

To finish this project, we need to solve two subproblems.

1. The Vehicle Model
2. NMPC (How to define a optimization probelm)

The vehicle model can predict the car's further states (location, speed, orientation, etc) according to the current state and actuators. 

Once we get the future trajectory which is in the future states, we can compare it with our target trajectory and calculate the loss between them. 

Finally we can choose actuators which minimize the loss before. So we get the steering angle, throttle, and brake value.

## The Vehicle Model

The vehicle model used in this project is a kinematic bicycle model which ignores tire forces, gravity and mass. At low and moderate speeds, kinematic models often approximate the actual vehicle dynamics. Kinematic model is implemented using the following equations:

```
x_[t+1] = x[t] + v[t] * cos(psi[t]) * dt
y_[t+1] = y[t] + v[t] * sin(psi[t]) * dt
psi_[t+1] = psi[t] + v[t] / Lf * delta[t] * dt
v_[t+1] = v[t] + a[t] * dt
cte[t+1] = f(x[t]) - y[t] + v[t] * sin(epsi[t]) * dt
epsi[t+1] = psi[t] - psides[t] + v[t] * delta[t] / Lf * dt
```

Here, `x,y` denote the position of the car, `psi` the heading direction, `v` its velocity `cte` the cross-track error and `epsi` the orientation error. `Lf` is the distance between the center of mass of the vehicle and the front wheels.

## NMPC

The NMPC algorithms contain three parts:

1. Input: car's current state and target trajectory
2. Optimization: build loss, variables and constrains
3. Output: the variables which minimize the loss

The overall frame can been seen in the follow two pictures:

![这里写图片描述](http://img.blog.csdn.net/20171209151214517?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvWW91bmdfR3k=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

![这里写图片描述](http://img.blog.csdn.net/20171209151229428?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvWW91bmdfR3k=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

### Coordinates Transformation

Before the mpc, we need to preprocess the values provided by the simulator. The values are in the global system. To make things easier, we need to transform them to car's coordinates. The transmission is:

```
X' =   cos(psi) * (ptsx[i] - x) + sin(psi) * (ptsy[i] - y);
Y' =  -sin(psi) * (ptsx[i] - x) + cos(psi) * (ptsy[i] - y);  
```

The initial state in the cars's coordinates is:

```
state << 0, 0, 0, v, cte, epsi;
```

### Timestep Length and Elapsed Duration (N & dt)

The time `T=N dt` defines the prediction horizon. Short prediction horizon makes the controller more responsive but suffers form instability. Long prediction horizon makes the car look further and stabilize car's trajectory. However, the car may not respond in time facing a sharp curve.

Here I chose values of `N` and `dt` manually such that the car run smoothly around 70mph. The final values are: `N = 10`, `dt = 0.05`.


### Cost Function

The loss of a trajectory of length N is computed as follows:

```
Cost  = Sum_i cte(i)^2 
    + epsi(i)^2 
    + (v(i)-v_ref)^2 
    + 200 delta(i)^2 
    + a(i)^2 
    + 600 [delta(i+1)-delta(i)] 
    + 100 [a(i+1)-a(i)]
```

### NMPC with Latency

The delay make the control problem a so-called sampled NMPC problem. When delays are not properly dealt with, the car will run with  oscillations.

I solve the latency problem by restrain controls to the values of the previous iteration for the duration of latency. I get the actual control values with the delay of latency time.


## Dependencies

* cmake >= 3.5
 * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1(mac, linux), 3.81(Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools]((https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)
* [uWebSockets](https://github.com/uWebSockets/uWebSockets)
  * Run either `install-mac.sh` or `install-ubuntu.sh`.
  * If you install from source, checkout to commit `e94b6e1`, i.e.
    ```
    git clone https://github.com/uWebSockets/uWebSockets
    cd uWebSockets
    git checkout e94b6e1
    ```
    Some function signatures have changed in v0.14.x. See [this PR](https://github.com/udacity/CarND-MPC-Project/pull/3) for more details.

* **Ipopt and CppAD:** Please refer to [this document](https://github.com/udacity/CarND-MPC-Project/blob/master/install_Ipopt_CppAD.md) for installation instructions.
* [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page). This is already part of the repo so you shouldn't have to worry about it.
* Simulator. You can download these from the [releases tab](https://github.com/udacity/self-driving-car-sim/releases).
* Not a dependency but read the [DATA.md](./DATA.md) for a description of the data sent back from the simulator.


## Basic Build Instructions

1. Clone this repo.
2. Make a build directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./mpc`.