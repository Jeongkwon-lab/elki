package de.lmu.ifi.dbs.elki.distance.distancefunction.minkowski;

/*
 This file is part of ELKI:
 Environment for Developing KDD-Applications Supported by Index-Structures

 Copyright (C) 2014
 Ludwig-Maximilians-Universität München
 Lehr- und Forschungseinheit für Datenbanksysteme
 ELKI Development Team

 This program is free software: you can redistribute it and/or modify
 it under the terms of the GNU Affero General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU Affero General Public License for more details.

 You should have received a copy of the GNU Affero General Public License
 along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

import java.util.Arrays;

import de.lmu.ifi.dbs.elki.data.NumberVector;
import de.lmu.ifi.dbs.elki.data.spatial.SpatialComparable;

/**
 * Provides the Euclidean distance for FeatureVectors.
 * 
 * @author Erich Schubert
 */
public class WeightedEuclideanDistanceFunction extends WeightedLPNormDistanceFunction {
  /**
   * Constructor.
   * 
   * @param weights
   */
  public WeightedEuclideanDistanceFunction(double[] weights) {
    super(2.0, weights);
  }

  private final double preDistance(NumberVector v1, NumberVector v2, final int start, final int end, double agg) {
    for(int d = start; d < end; d++) {
      final double xd = v1.doubleValue(d), yd = v2.doubleValue(d);
      final double delta = xd - yd;
      agg += delta * delta * weights[d];
    }
    return agg;
  }

  private final double preDistanceVM(NumberVector v, SpatialComparable mbr, final int start, final int end, double agg) {
    for(int d = start; d < end; d++) {
      final double value = v.doubleValue(d), min = mbr.getMin(d);
      double delta = min - value;
      if(delta < 0.) {
        delta = value - mbr.getMax(d);
      }
      if(delta > 0.) {
        agg += delta * delta * weights[d];
      }
    }
    return agg;
  }

  private final double preDistanceMBR(SpatialComparable mbr1, SpatialComparable mbr2, final int start, final int end, double agg) {
    for(int d = start; d < end; d++) {
      double delta = mbr2.getMin(d) - mbr1.getMax(d);
      if(delta < 0.) {
        delta = mbr1.getMin(d) - mbr2.getMax(d);
      }
      if(delta > 0.) {
        agg += delta * delta * weights[d];
      }
    }
    return agg;
  }

  private final double preNorm(NumberVector v, final int start, final int end, double agg) {
    for(int d = start; d < end; d++) {
      final double xd = v.doubleValue(d);
      agg += xd * xd * weights[d];
    }
    return agg;
  }

  private final double preNormMBR(SpatialComparable mbr, final int start, final int end, double agg) {
    for(int d = start; d < end; d++) {
      double delta = mbr.getMin(d);
      if(delta < 0.) {
        delta = -mbr.getMax(d);
      }
      if(delta > 0.) {
        agg += delta * delta * weights[d];
      }
    }
    return agg;
  }

  @Override
  public double distance(NumberVector v1, NumberVector v2) {
    final int dim1 = v1.getDimensionality(), dim2 = v2.getDimensionality();
    final int mindim = (dim1 < dim2) ? dim1 : dim2;
    double agg = preDistance(v1, v2, 0, mindim, 0.);
    if(dim1 > mindim) {
      agg = preNorm(v1, mindim, dim1, agg);
    }
    else if(dim2 > mindim) {
      agg = preNorm(v2, mindim, dim2, agg);
    }
    return Math.sqrt(agg);
  }

  @Override
  public double norm(NumberVector v) {
    return Math.sqrt(preNorm(v, 0, v.getDimensionality(), 0.));
  }

  @Override
  public double minDist(SpatialComparable mbr1, SpatialComparable mbr2) {
    final int dim1 = mbr1.getDimensionality(), dim2 = mbr2.getDimensionality();
    final int mindim = (dim1 < dim2) ? dim1 : dim2;

    final NumberVector v1 = (mbr1 instanceof NumberVector) ? (NumberVector) mbr1 : null;
    final NumberVector v2 = (mbr2 instanceof NumberVector) ? (NumberVector) mbr2 : null;

    double agg = 0.;
    if(v1 != null) {
      if(v2 != null) {
        agg = preDistance(v1, v2, 0, mindim, agg);
      }
      else {
        agg = preDistanceVM(v1, mbr2, 0, mindim, agg);
      }
    }
    else {
      if(v2 != null) {
        agg = preDistanceVM(v2, mbr1, 0, mindim, agg);
      }
      else {
        agg = preDistanceMBR(mbr1, mbr2, 0, mindim, agg);
      }
    }
    // first object has more dimensions.
    if(dim1 > mindim) {
      if(v1 != null) {
        agg = preNorm(v1, mindim, dim1, agg);
      }
      else {
        agg = preNormMBR(v1, mindim, dim1, agg);
      }
    }
    // second object has more dimensions.
    if(dim2 > mindim) {
      if(v2 != null) {
        agg = preNorm(v2, mindim, dim2, agg);
      }
      else {
        agg = preNormMBR(mbr2, mindim, dim2, agg);
      }
    }
    return Math.sqrt(agg);
  }

  @Override
  public boolean equals(Object obj) {
    if(this == obj) {
      return true;
    }
    if(obj == null) {
      return false;
    }
    if(!(obj instanceof WeightedEuclideanDistanceFunction)) {
      if(obj.getClass().equals(WeightedLPNormDistanceFunction.class)) {
        return super.equals(obj);
      }
      if(obj.getClass().equals(EuclideanDistanceFunction.class)) {
        for(double d : weights) {
          if(d != 1.0) {
            return false;
          }
        }
        return true;
      }
      return false;
    }
    WeightedEuclideanDistanceFunction other = (WeightedEuclideanDistanceFunction) obj;
    return Arrays.equals(this.weights, other.weights);
  }
}
