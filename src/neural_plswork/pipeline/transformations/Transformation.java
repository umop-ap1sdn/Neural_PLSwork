package neural_plswork.pipeline.transformations;

import neural_plswork.dataframe.DoubleColumn;

public interface Transformation {
    
    public default DoubleColumn[] fit_transform(DoubleColumn[] input) {
        fit(input);
        return transform(input);
    }

    public void fit(DoubleColumn[] input);
    public DoubleColumn[] transform(DoubleColumn[] input);
}
