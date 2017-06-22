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

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	default_random_engine generator;
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  num_particles = 100;
  for (int i = 0; i < num_particles; i++) {
    Particle p;
    p.id = i;
    p.x = dist_x(generator);
    p.y = dist_y(generator);
    p.theta = dist_theta(generator);
    p.weight = 1;
    particles.push_back(p);
    weights.push_back(p.weight);
  }
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
  default_random_engine generator;

  for (int i = 0; i < num_particles; i++) {
    double x_1 = 0;
    double y_1 = 0;
    double theta_1 = 0;
    Particle p = particles[i];

    if (abs(yaw_rate) <= 0.0001) {
      x_1 = p.x + velocity * delta_t * cos(p.theta);
      y_1 = p.y + velocity * delta_t * sin(p.theta);
      theta_1 = p.theta;
    } else {
      x_1 = p.x + velocity/yaw_rate * (sin(p.theta + yaw_rate*delta_t) - sin(p.theta));
      y_1 = p.y + velocity/yaw_rate * (cos(p.theta) - cos(p.theta + yaw_rate*delta_t));
      theta_1 = p.theta + yaw_rate*delta_t;
    }
    normal_distribution<double> dist_x(x_1, std_pos[0]);
    normal_distribution<double> dist_y(y_1, std_pos[1]);
    normal_distribution<double> dist_theta(theta_1, std_pos[2]);

    p.x = dist_x(generator);
    p.y = dist_y(generator);
    p.theta = dist_theta(generator);

    particles.push_back(p);
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
  for (auto& obs : observations) {
    int min_dist = 100000; //TODO replace with max int
    for (auto pred : predicted) {
      int distance = dist(obs.x, obs.y, pred.x, pred.y);
      if (distance < min_dist) {
        obs.id = pred.id;
        min_dist = distance;
      }
    }
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
  for (auto& p : particles) {
    vector<LandmarkObs> transformed_obs;
    vector<LandmarkObs> landmarks;

    // TODO: filter out landmarks outside of sensor range
    for (auto map_landmark : map_landmarks.landmark_list) {
      LandmarkObs map_landmark_obs = {map_landmark.id_i, map_landmark.x_f, map_landmark.y_f};
      landmarks.push_back(map_landmark_obs);
    }

    for (auto obs : observations) {
      LandmarkObs t_obs;
      t_obs.x = p.x + obs.x*cos(p.theta) - obs.y*sin(p.theta);
      t_obs.y = p.y + obs.x*sin(p.theta) + obs.y*cos(p.theta);
      transformed_obs.push_back(t_obs);
    }

    dataAssociation(landmarks, transformed_obs);

    p.weight = 1;

    for (auto obs : transformed_obs)
    {
      double mu_x, mu_y;
      double sigma_x = std_landmark[0];
      double sigma_y = std_landmark[1];
      double x = obs.x;
      double y = obs.y;

      //TODO use hashmap to improve efficiency
      for (auto landmark : landmarks) {
        if (landmark.id == obs.id) {
          mu_x = landmark.x;
          mu_y = landmark.y;
        }
      }

      double constant_factor = 1 / (2 * M_PI * sigma_x * sigma_y);
      double x_term = (x - mu_x) * (x - mu_x) / (2 * sigma_x * sigma_x);
      double y_term = (y - mu_y) * (y - mu_y) / (2 * sigma_y * sigma_y);

      p.weight *= constant_factor * exp(-(x_term + y_term));
      cout << p.weight<<endl;
    }
    weights.push_back(p.weight);
  }
}

void ParticleFilter::resample() {
  default_random_engine generator;
  discrete_distribution<int> disc_dist(weights.begin(), weights.end());

  vector<Particle> new_particles;
  for (int i = 0; i < num_particles; i++) {
    Particle new_p = particles[disc_dist(generator)];
    new_particles.push_back(new_p);
  }
  particles = new_particles;
  weights.clear();
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
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
