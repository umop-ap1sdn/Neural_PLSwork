package neural_plswork.optimizer;

import neural_plswork.core.NetworkValue;
import neural_plswork.math.Matrix;

public class Adam implements OptimizationFunction {

    private Matrix<NetworkValue> m_t;
    private Matrix<NetworkValue> v_t;

    private final double BETA_1;
    private final double BETA_2;
    private final double EPSILON;

    int iteration;

    public Adam(double BETA_1, double BETA_2, double EPSILON) {
        this.BETA_1 = BETA_1;
        this.BETA_2 = BETA_2;
        this.EPSILON = EPSILON;

        iteration = 0;
        m_t = null;
        v_t = null;
    }

    protected Adam() {
        this.BETA_1 = 0.9;
        this.BETA_2 = 0.999;
        this.EPSILON = 1e-7;

        iteration = 0;
        m_t = null;
        v_t = null;
    }
    
    @Override
    public Matrix<NetworkValue> computeDeltas(Matrix<NetworkValue> gradients, double learning_rate) {
        Matrix<NetworkValue> g2 = gradients.copy();
        for(NetworkValue n: g2) {
            n.setValue(Math.pow(n.getValue(), 2));
        }


    }

    

    @Override
    public OptimizationFunction copy() {
        return new Adam(BETA_1, BETA_2, EPSILON);
    }
    
}
