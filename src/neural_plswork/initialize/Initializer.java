package neural_plswork.initialize;

import neural_plswork.core.NetworkValue;

public interface Initializer {
    public NetworkValue getNextWeight(int row, int column);
}
