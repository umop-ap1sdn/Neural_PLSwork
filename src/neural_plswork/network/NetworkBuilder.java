package neural_plswork.network;

import neural_plswork.evaluation.Differentiable;
import neural_plswork.evaluation.Evaluation;
import neural_plswork.evaluation.loss.Loss;
import neural_plswork.evaluation.loss.LossFunction;
import neural_plswork.initialize.Initializer;
import neural_plswork.initialize.WeightInitializer;
import neural_plswork.optimizer.OptimizationFunction;
import neural_plswork.optimizer.Optimizer;
import neural_plswork.regularization.penalize.Penalty;
import neural_plswork.regularization.penalize.WeightPenalizer;

public class NetworkBuilder {
    private static final Initializer DEFAULT_INITIALIZER = Initializer.getPenalty(WeightInitializer.UNIF);
    private static final Penalty DEFAULT_PENALTY = Penalty.getPenalty(WeightPenalizer.NONE, 0, 0);
    private static final OptimizationFunction DEFAULT_OPTIMIZER = OptimizationFunction.getFunction(Optimizer.SGD);
    private final Differentiable DEFAULT_TRAINING_EVALUATOR;
    private final Evaluation DEFAULT_REPORTING_EVALUATOR;

    private final int MAX_THREADS;
    private final int BATCH_SIZE;

    public NetworkBuilder(int MAX_THREADS, int BATCH_SIZE) {
        this.MAX_THREADS = MAX_THREADS;
        this.BATCH_SIZE = BATCH_SIZE;

        DEFAULT_TRAINING_EVALUATOR = (Differentiable) LossFunction.getFunction(Loss.MSE, BATCH_SIZE);
        DEFAULT_REPORTING_EVALUATOR = LossFunction.getFunction(Loss.MSE, BATCH_SIZE);
    }
}
