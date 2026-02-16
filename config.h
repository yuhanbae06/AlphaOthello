#pragma once
#include <fstream>
#include <iostream>

#include "json.hpp"

using json = nlohmann::json;

struct Config {
  int C;
  int num_searches;
  int num_selfPlay_iterations;
  int num_parallel_games;
  float temperature;
  float dirichlet_epsilon;
  float dirichlet_alpha;

  static Config load() {
    std::ifstream f("config.json");

    json j;
    f >> j;

    Config c;

    c.C = j.value("C", 2);
    c.num_searches = j.value("num_searches", 1);
    c.num_selfPlay_iterations = j.value("num_selfPlay_iterations", 5);
    c.num_parallel_games = j.value("num_parallel_games", 1);
    c.temperature = j.value("temperature", 1.25f);
    c.dirichlet_epsilon = j.value("dirichlet_epsilon", 0.25f);
    c.dirichlet_alpha = j.value("dirichlet_alpha", 0.03f);

    return c;
  }
};