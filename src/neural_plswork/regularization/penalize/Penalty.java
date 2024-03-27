package neural_plswork.regularization.penalize;

import neural_plswork.core.NetworkValue;
import neural_plswork.math.Matrix;

public interface Penalty {
    public Matrix<NetworkValue> getPenalty(Matrix<NetworkValue> weights);
    public Matrix<NetworkValue> getDerivative(Matrix<NetworkValue> weights);

    public static Penalty getPenalty(WeightPenalizer penalty, double l1, double l2) throws InvalidPenaltyException {
        switch(penalty) {
            case CUSTOM: return null;
            case NONE: return new None();
            case LASSO: return new Lasso(l1);
            case RIDGE: return new Ridge(l2);
            case ELASTIC: return new Elastic(l1, l2);
            case INVALID: throw new InvalidPenaltyException();
            default: throw new InvalidPenaltyException();
        }
    }
}
