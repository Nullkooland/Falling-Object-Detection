#pragma once

class GravityProjector final {
  public:
    GravityProjector(float g = 9.80665F);
  private:
    float _g;
};