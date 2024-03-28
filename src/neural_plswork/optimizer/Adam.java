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
        
        Matrix<NetworkValue> m_update = gradients.scale(new NetworkValue(1 - BETA_1));
        Matrix<NetworkValue> v_update = g2.scale(new NetworkValue(1 - BETA_2));

        if(m_t == null) m_t = m_update;
        else m_t = m_t.scale(new NetworkValue(BETA_1)).add(m_update);

        if(v_t == null) v_t = v_update;
        else v_t = v_t.scale(new NetworkValue(BETA_2)).add(v_update);

        Matrix<NetworkValue> m_hat = m_t.scale(new NetworkValue(1.0 / (1.0 - Math.pow(BETA_1, iteration))));
        Matrix<NetworkValue> v_hat = v_t.scale(new NetworkValue(1.0 / (1.0 - Math.pow(BETA_2, iteration))));

        Matrix<NetworkValue> ret = new Matrix<>(m_hat.getRows(), m_hat.getColumns());

        for(int i = 0; i < m_hat.getRows(); i++) {
            for(int j = 0; j < v_hat.getColumns(); j++) {
                double value = learning_rate * m_hat.getValue(i, j).getValue() / (Math.sqrt(v_hat.getValue(i, j).getValue()) + EPSILON);
                ret.setValue(new NetworkValue(value), i, j);
            }
        }
        

        return ret;

    }

    

    @Override
    public OptimizationFunction copy() {
        return new Adam(BETA_1, BETA_2, EPSILON);
    }
    
}
