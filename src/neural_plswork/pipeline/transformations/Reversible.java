package neural_plswork.pipeline.transformations;

import neural_plswork.dataframe.DoubleColumn;

public interface Reversible {
    public DoubleColumn[] reverse(DoubleColumn[] input);
}
