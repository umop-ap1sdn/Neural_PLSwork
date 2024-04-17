package neural_plswork.neuron.dropout;

import java.util.Arrays;

public class NoneDropout implements Dropout {

    @Override
    public boolean[] dropout(int length) {
        boolean[] ret = new boolean[length];
        Arrays.fill(ret, false);
        return ret;
    }

    @Override
    public Dropout copy() {
        return new NoneDropout();
    }
    
}
