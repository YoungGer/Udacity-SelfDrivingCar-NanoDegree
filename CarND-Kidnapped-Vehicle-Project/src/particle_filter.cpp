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

    num_particles = 1000;

    /*
     * Set particles
     */
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

    /*
     * Set others
     */
    weights.assign(num_particles, 1.0);
    is_initialized = true;

    cout<<"Init Finished"<<endl;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

    cout<<"Prediction Start"<<endl;

    for (int i = 0; i < num_particles; ++i) {
        // main
        if (yaw_rate>0.0001){
            double theta0 = particles[i].theta;
            particles[i].x += velocity/yaw_rate * ( sin(theta0+yaw_rate*delta_t) - sin(theta0) );
            particles[i].y += velocity/yaw_rate * ( -cos(theta0+yaw_rate*delta_t) + cos(theta0) );
            particles[i].theta += theta0 * delta_t;
        } else {
            particles[i].x += velocity * delta_t * cos(particles[i].theta);
            particles[i].y += velocity * delta_t * sin(particles[i].theta);
        }
        // add random noise
        double std_x = std_pos[0];
        double std_y = std_pos[1];
        double std_theta = std_pos[2];
        normal_distribution<double> dist_x(particles[i].x, std_x);
        normal_distribution<double> dist_y(particles[i].y, std_y);
        normal_distribution<double> dist_theta(particles[i].theta, std_theta);
        particles[i].x = dist_x(gen);
        particles[i].y = dist_y(gen);
        particles[i].theta = dist_theta(gen);
    }

    cout<<"Prediction Finished"<<endl;
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

    //

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
    cout<<"Update Start"<<endl;
    // iterate particles
    weights.clear();
    for (int i = 0; i < num_particles; ++i) {
        double x_p = particles[i].x;
        double y_p = particles[i].y;
        double theta = particles[i].theta;

        // clear
        particles[i].associations.clear();
        particles[i].sense_x.clear();
        particles[i].sense_y.clear();

        // iterate obes
        double weight_p = 1.0;
        for (int j = 0; j < observations.size(); ++j) {
            double x_o = observations[i].x;
            double y_o = observations[i].y;

            // coordinate system change  from VEHICLE'S to MAP'S
            double x_m, y_m;
            x_m = x_p + cos(theta)*x_o - sin(theta)*y_o;
            y_m = y_p + sin(theta)*x_o + cos(theta)*y_o;

            // choose landmarks which has the distance with particle less than sensor_range
            vector<Map::single_landmark_s> effect_landmark_list;
            for (int k=0; k < map_landmarks.landmark_list.size(); ++k) {
                float x_l_i = map_landmarks.landmark_list[i].x_f;
                float y_l_i = map_landmarks.landmark_list[i].y_f;
                double dist = sqrt( pow(x_p-x_l_i, 2) + pow(y_p-y_l_i, 2) );
                if (dist < sensor_range){
                    effect_landmark_list.push_back(map_landmarks.landmark_list[i]);
                }
            }

            // if cannot find associate landmake
            if (effect_landmark_list.size()==0){
                particles[i].associations.push_back(0);
                particles[i].sense_x.push_back(x_m);
                particles[i].sense_y.push_back(y_m);
                weight_p = 0;
                continue;
            }

            // iterate landmarks to allocate (x_m,y_m) with (x_landmark, y_landmark)
            float x_l, y_l;
            int id_l;
            double min_diff = DBL_MAX;
            for (int k=0; k < effect_landmark_list.size(); ++k) {
                float x_l_i = effect_landmark_list[i].x_f;
                float y_l_i = effect_landmark_list[i].y_f;
                int id_l_i = effect_landmark_list[i].id_i;

                if ( pow(x_l_i-x_m, 2) + pow(y_l_i-y_m, 2)  < min_diff ) {
                    id_l = id_l_i;
                    x_l = x_l_i;
                    y_l = y_l_i;
                }
            }

            // calculate weight with this particle and this landmark
            double std_x = std_landmark[0];
            double std_y = std_landmark[1];
            double p_one_landmark = 1/(2*M_PI*std_x*std_y)*exp(-( pow(x_m-x_l,2)/(2*std_x*std_x) + pow(y_m-y_l,2)/(2*std_y*std_y) ));

            // update
            weight_p *= p_one_landmark;
            particles[i].associations.push_back(id_l);
            particles[i].sense_x.push_back(x_m);
            particles[i].sense_y.push_back(y_m);
        }

        // update particle
        particles[i].weight = weight_p;
        weights.push_back(weight_p);
    }


    cout<<"Update Finished"<<endl;
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    cout<<"Resample Start"<<endl;
    discrete_distribution<int> index (weights.begin(), weights.end());

    std::vector<Particle> new_particles;
    for (int i = 0; i < num_particles; ++i) {
        new_particles.push_back(particles[index(gen)]);
    }
    particles = new_particles;

    cout<<"Resample Finished"<<endl;
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
