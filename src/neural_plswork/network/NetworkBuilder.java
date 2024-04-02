package neural_plswork.network;

import java.util.ArrayList;

import neural_plswork.evaluation.Differentiable;
import neural_plswork.evaluation.Evaluation;
import neural_plswork.evaluation.loss.Loss;
import neural_plswork.evaluation.loss.LossFunction;
import neural_plswork.initialize.Initializer;
import neural_plswork.initialize.WeightInitializer;
import neural_plswork.layers.basic.InputLayer;
import neural_plswork.optimizer.OptimizationFunction;
import neural_plswork.optimizer.Optimizer;
import neural_plswork.regularization.penalize.Penalty;
import neural_plswork.regularization.penalize.WeightPenalizer;
import neural_plswork.unit.Unit;
import neural_plswork.unit.ffUnits.OutputUnit;

public class NetworkBuilder {
    private static final Initializer DEFAULT_INITIALIZER = Initializer.getPenalty(WeightInitializer.UNIF);
    private static final Penalty DEFAULT_PENALTY = Penalty.getPenalty(WeightPenalizer.NONE, 0, 0);
    private static final OptimizationFunction DEFAULT_OPTIMIZER = OptimizationFunction.getFunction(Optimizer.SGD);
    private final Differentiable DEFAULT_TRAINING_EVALUATOR;
    private final Evaluation DEFAULT_REPORTING_EVALUATOR;

    private Penalty penalty;
    private Differentiable evaluator;
    private Evaluation reporter;

    private final int MAX_THREADS;
    private final int BATCH_SIZE;

    private InputLayer input;
    private ArrayList<Unit> hidden;
    private OutputUnit output;

    public NetworkBuilder(int MAX_THREADS, int BATCH_SIZE) {
        this.MAX_THREADS = MAX_THREADS;
        this.BATCH_SIZE = BATCH_SIZE;

        DEFAULT_TRAINING_EVALUATOR = (Differentiable) LossFunction.getFunction(Loss.MSE, BATCH_SIZE);
        DEFAULT_REPORTING_EVALUATOR = LossFunction.getFunction(Loss.MSE, BATCH_SIZE);

        this.penalty = DEFAULT_PENALTY;
        this.evaluator = DEFAULT_TRAINING_EVALUATOR;
        this.reporter = DEFAULT_REPORTING_EVALUATOR;

        hidden = new ArrayList<>();
    }

    public void setPenalty(WeightPenalizer penalizer, double l1, double l2) {
        this.penalty = Penalty.getPenalty(penalizer, l1, l2);
    }

    public void setEvaluator(Loss loss) throws InvalidNetworkConstructionException {
        try {
            this.evaluator = (Differentiable) LossFunction.getFunction(loss, BATCH_SIZE);
        } catch (ClassCastException e) {
            throw new InvalidNetworkConstructionException("Evaluation function must be Differentiable");
        }
    }

    public void setEvaluator(Differentiable eval) {
        this.evaluator = eval;
    }

    public void setReporter(Loss loss) {
        this.reporter = LossFunction.getFunction(loss, BATCH_SIZE);
    }

    public void setReporter(Evaluation eval) {
        this.reporter = eval;
    }
}
