/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <float.h>
#include <stdio.h>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    // TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
    //   x, y, theta and their uncertainties from GPS) and all weights to 1.
    // Add random Gaussian noise to each particle.
    // NOTE: Consult particle_filter.h for more information about this method (and others in this file).
    cout<<"Init Start"<<endl;

    num_particles = 100;


    /*******************************************************
     * Set particles
     ******************************************************/
    double std_x = std[0];
    double std_y = std[1];
    double std_theta = std[2];

    // set random engine
    normal_distribution<double> dist_x(x, std_x);
    normal_distribution<double> dist_y(y, std_y);
    normal_distribution<double> dist_theta(theta, std_theta);

    for (int i = 0; i < num_particles; ++i) {
        Particle curr_particle;
        curr_particle.id = i;
        curr_particle.x = dist_x(gen);
        curr_particle.y = dist_y(gen);
        curr_particle.theta = dist_theta(gen);
        curr_particle.weight = 1.0;
        particles.push_back(curr_particle);
    }

    /******************************************************
     * Set others
     ******************************************************/
    weights.assign(num_particles, 1.0);
    is_initialized = true;

    cout<<"Weights Size: "<<weights.size()<<endl;
    cout<<"Particles Size: "<<particles.size()<<endl;
    cout<<"Init Finished"<<endl;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
    // TODO: Add measurements to each particle and add random Gaussian noise.
    // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
    //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
    //  http://www.cplusplus.com/reference/random/default_random_engine/


    double std_x = std_pos[0];
    double std_y = std_pos[1];
    double std_theta = std_pos[2];
    normal_distribution<double> dist_x(0, std_x);
    normal_distribution<double> dist_y(0, std_y);
    normal_distribution<double> dist_theta(0, std_theta);


    for (int i = 0; i < num_particles; ++i) {
        // main
        if (fabs(yaw_rate)>0.001){
            double theta0 = particles[i].theta;
            particles[i].x += velocity/yaw_rate * ( sin(theta0+yaw_rate*delta_t) - sin(theta0) );
            particles[i].y += velocity/yaw_rate * ( -cos(theta0+yaw_rate*delta_t) + cos(theta0) );
            particles[i].theta += yaw_rate * delta_t;
        } else {
            particles[i].x += velocity * delta_t * cos(particles[i].theta);
            particles[i].y += velocity * delta_t * sin(particles[i].theta);
        }
        // add random noise
        particles[i].x += dist_x(gen);
        particles[i].y += dist_y(gen);
        particles[i].theta += dist_theta(gen);
    }

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
    // TODO: Find the predicted measurement that is closest to each observed measurement and assign the
    //   observed measurement to this particular landmark.
    // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
    //   implement this method and use it as a helper during the updateWeights phase.


    const double BIG_NUMBER = 1.0e99;

    for (int i = 0; i < observations.size(); i++) {

        int current_j, id_j;
        double current_smallest_error = BIG_NUMBER;

        for (int j = 0; j < predicted.size(); j++) {

            const double dx = predicted[j].x - observations[i].x;
            const double dy = predicted[j].y - observations[i].y;
            const int id_j = predicted[j].id;
            const double error = dx * dx + dy * dy;

            if (error < current_smallest_error) {
                current_j = j;
                current_smallest_error = error;
            }
        }
        observations[i].ii = current_j;
        observations[i].id = id_j;
    }

}



void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
    // TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
    //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
    // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
    //   according to the MAP'S coordinate system. You will need to transform between the two systems.
    //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
    //   The following is a good resource for the theory:
    //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
    //   and the following is a good resource for the actual equation to implement (look at equation
    //   3.33
    //   http://planning.cs.uiuc.edu/node99.html

    // iterate particles
    weights.clear();
    for (int i = 0; i < num_particles; ++i) {

        // get particle infomation
        double x_p = particles[i].x;
        double y_p = particles[i].y;
        double theta = particles[i].theta;

        // clear
        particles[i].associations.clear();
        particles[i].sense_x.clear();
        particles[i].sense_y.clear();

        /*****************************************************
         * Step 1: coordinate system change from VEHICLE'S to MAP'S
         ****************************************************/
        vector<LandmarkObs> map_observations;
        for (int j = 0; j < observations.size(); ++j) {
            double x_o = observations[j].x;
            double y_o = observations[j].y;

            double x_m, y_m;
            x_m = x_p + cos(theta) * x_o - sin(theta) * y_o;
            y_m = y_p + sin(theta) * x_o + cos(theta) * y_o;

            LandmarkObs observation = {
                    observations[j].id,
                    0,
                    x_m,
                    y_m
            };

            map_observations.push_back(observation);
        }

        /*****************************************************
         * Step 2: choose landmarks which has the distance with particle less than sensor_range
         ****************************************************/
        vector<LandmarkObs> effect_landmark_list;
        for (int k=0; k < map_landmarks.landmark_list.size(); ++k) {
            float x_l_i = map_landmarks.landmark_list[k].x_f;
            float y_l_i = map_landmarks.landmark_list[k].y_f;
            int id_l_i = map_landmarks.landmark_list[k].id_i;
            double dist = sqrt( pow(x_p-x_l_i, 2) + pow(y_p-y_l_i, 2) );
            if (dist < sensor_range){
                LandmarkObs curr_landmark = {
                        id_l_i,
                        0,
                        x_l_i,
                        y_l_i
                };
                effect_landmark_list.push_back(curr_landmark);
            }
        }


        /*****************************************************
         * Step 3: associate landmarks in range to landmarks obeservations
         ****************************************************/
        ParticleFilter::dataAssociation(effect_landmark_list, map_observations);

        /*****************************************************
         * Step 4: update weight
         ****************************************************/
        double weight_p = 1.0;
        if (effect_landmark_list.size()==0){
            weight_p = 0.0;
        } else{
            for (int j = 0; j < map_observations.size(); ++j) {
                // obeseration im map coordinate
                double x_m = map_observations[j].x;
                double y_m = map_observations[j].y;
                // landmark im map coordinate
                double x_l = effect_landmark_list[map_observations[j].ii].x;
                double y_l = effect_landmark_list[map_observations[j].ii].y;

                double std_x = std_landmark[0];
                double std_y = std_landmark[1];
                double p_one_landmark = 1/(2*M_PI*std_x*std_y)*exp(-( pow(x_m-x_l,2)/(2*std_x*std_x) + pow(y_m-y_l,2)/(2*std_y*std_y) ));

                weight_p *= p_one_landmark;

                particles[i].associations.push_back(map_observations[j].id);

                // !!!!! I need help in follow two lines, please help me!!!!!
                // !!!!! I cannot run my program when I uncomment the follow two lines. Why and How can I solve it?
                //particles[i].sense_x.push_back(x_m);
                //particles[i].sense_y.push_back(y_m);
            }
        }

        // update particle
        particles[i].weight = weight_p;
        weights.push_back(weight_p);
    }
}

void ParticleFilter::resample() {
    // TODO: Resample particles with replacement with probability proportional to their weight.
    // NOTE: You may find std::discrete_distribution helpful here.
    //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    discrete_distribution<int> index (weights.begin(), weights.end());

    std::vector<Particle> new_particles;
    for (int i = 0; i < num_particles; ++i) {
        new_particles.push_back(particles[index(gen)]);
    }
    particles = new_particles;

}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations,
                                         const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
    vector<int> v = best.associations;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
    vector<double> v = best.sense_x;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
    vector<double> v = best.sense_y;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
