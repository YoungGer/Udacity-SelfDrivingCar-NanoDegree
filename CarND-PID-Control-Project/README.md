# CarND-Controls-PID

This is a Self-Driving Car Engineer Nanodegree Program completed by Guangyao Yang. This program implements a PID controllor for controling a vehichle.


---


## Basic And Run

1. Clone this repo.
2. Make a build directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./pid`. 

## Reflection

The different effects of PID components:

- P component: It represents the respond strength to the error. For example, if the P component is big, the car can steer heavily to the right even it's slightly on the left. However, if the P component is small, the car can respond slowly so that it cannot fix its wrong orientation in time. As a result, the car's path can oscillate if there is only P component.
- I component: It represents the compensate to systematic bias. For example, if the steering of the car has been wrongly fixed and has a tendency to steer a bit to the left. Even if we set steer to zero, the car can steer to left due to the systematic bias. That's where the I component take effect. It accumulates systematic errors and amends the errors.
- D component: It represents the counter strength to the P component. The pure P component can easily lead to overshooting and oscillation. With the help of D component which decreases as the car gets closer to the target path, the car can avoid the over-shooting previously caused by the P controller alone.

Final parameters are tuned by hand. The parameters are:

- P: 0.2
- I: 0.0
- D: 5.0



## Dependencies

* cmake >= 3.5

* make >= 4.1(mac, linux), 3.81(Windows)

* gcc/g++ >= 5.4

* [uWebSockets](https://github.com/uWebSockets/uWebSockets)
  * Run either `./install-mac.sh` or `./install-ubuntu.sh`.
  * If you install from source, checkout to commit `e94b6e1`, i.e.
    ```
    git clone https://github.com/uWebSockets/uWebSockets 
    cd uWebSockets
    git checkout e94b6e1
    ```

* Simulator. You can download these from the [project intro page](https://github.com/udacity/self-driving-car-sim/releases) in the classroom.


