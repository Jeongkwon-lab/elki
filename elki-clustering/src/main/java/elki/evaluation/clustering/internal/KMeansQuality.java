package elki.evaluation.clustering.internal;

import java.util.List;

import elki.clustering.kmeans.quality.BayesianInformationCriterion;
import elki.clustering.kmeans.quality.KMeansQualityMeasure;
import elki.data.Cluster;
import elki.data.Clustering;
import elki.data.NumberVector;
import elki.data.model.MeanModel;
import elki.data.model.Model;
import elki.database.Database;
import elki.database.relation.Relation;
import elki.distance.NumberVectorDistance;
import elki.distance.minkowski.EuclideanDistance;
import elki.evaluation.Evaluator;
import elki.logging.Logging;
import elki.logging.statistics.DoubleStatistic;
import elki.result.EvaluationResult;
import elki.result.EvaluationResult.MeasurementGroup;
import elki.result.Metadata;
import elki.result.ResultUtil;
import elki.utilities.exceptions.NotImplementedException;
import elki.utilities.optionhandling.OptionID;
import elki.utilities.optionhandling.Parameterizer;
import elki.utilities.optionhandling.parameterization.Parameterization;
import elki.utilities.optionhandling.parameters.ObjectParameter;

public class KMeansQuality implements Evaluator {
  /**
   * Logger for debug output.
   */
  private static final Logging LOG = Logging.getLogger(KMeansQuality.class);

  /**
   * Key for logging statistics.
   */
  private String key = KMeansQuality.class.getName();

  KMeansQualityMeasure<? super NumberVector> measure;

  NumberVectorDistance<? super NumberVector> distance;

  public KMeansQuality(KMeansQualityMeasure<NumberVector> measure, NumberVectorDistance<? super NumberVector> distance) {
    this.measure = measure;
    this.distance = distance;
  }

  public void evaluateClustering(Relation<NumberVector> rel, Clustering<?> cs) {
    boolean mm = true;
    for(Cluster<?> c : cs.getAllClusters()) {
      Model m = c.getModel();
      mm = mm & m instanceof MeanModel;
    }
    if(!mm) {
      throw new NotImplementedException();
    }
    Clustering<MeanModel> mcs = (Clustering<MeanModel>) cs;
    double res = measure.quality(mcs, distance, rel);
    if(LOG.isStatistics()) {
      LOG.statistics(new DoubleStatistic(key + "." + measure.getName(), res));
    }

    EvaluationResult ev = EvaluationResult.findOrCreate(cs, "Internal Clustering Evaluation");
    MeasurementGroup g = ev.findOrCreateGroup("KMeans-Quality");
    g.addMeasure(measure.getName(), res, 0, Double.POSITIVE_INFINITY, measure.lowerIsBetter());
    if(!Metadata.hierarchyOf(cs).addChild(ev)) {
      Metadata.of(ev).notifyChanged();
    }
  }

  @Override
  public void processNewResult(Object result) {
    List<Clustering<?>> crs = Clustering.getClusteringResults(result);
    if(crs.isEmpty()) {
      return;
    }
    Database db = ResultUtil.findDatabase(result);
    Relation<NumberVector> rel = db.getRelation(distance.getInputTypeRestriction());
    for(Clustering<?> c : crs) {
      evaluateClustering(rel, c);
    }
  }

  /**
   * Parameterization class.
   *
   * @author Andreas Lang
   */
  public static class Par implements Parameterizer {
    /**
     * Parameter for choosing the distance function.
     */
    public static final OptionID DISTANCE_ID = new OptionID("kmeansquality.distance", "Distance function to use.");

    /**
     * Parameter for choosing the distance function.
     */
    public static final OptionID MEASURE_ID = new OptionID("kmeansquality.measure", "measure function to use.");

    /**
     * Distance function to use.
     */
    private NumberVectorDistance<? super NumberVector> distance;

    KMeansQualityMeasure<NumberVector> measure;

    @Override
    public void configure(Parameterization config) {
      new ObjectParameter<NumberVectorDistance<? super NumberVector>>(DISTANCE_ID, NumberVectorDistance.class, EuclideanDistance.class) //
          .grab(config, x -> distance = x);
      new ObjectParameter<KMeansQualityMeasure<NumberVector>>(MEASURE_ID, KMeansQualityMeasure.class, BayesianInformationCriterion.class) //
          .grab(config, x -> measure = x);
    }

    @Override
    public KMeansQuality make() {
      return new KMeansQuality(measure, distance);
    }
  }
}
