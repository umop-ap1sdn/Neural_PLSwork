package neural_plswork.pipeline;

import java.util.ArrayList;

import neural_plswork.dataframe.DoubleColumn;
import neural_plswork.dataframe.NumericalDataFrame;
import neural_plswork.pipeline.transformations.Reversible;
import neural_plswork.pipeline.transformations.Transformation;

public class DataPipeline {
    private final DoubleColumn[] originalData;
    private DoubleColumn[] currentData;

    private final ArrayList<Transformation> path;
    private final ArrayList<Reversible> revPath;

    public DataPipeline(NumericalDataFrame ndf) {
        originalData = ndf.getData();
        currentData = originalData;

        path = new ArrayList<>();
        revPath = new ArrayList<>();
    }

    public DataPipeline(DoubleColumn[] data) {
        originalData = data;
        currentData = originalData;

        path = new ArrayList<>();
        revPath = new ArrayList<>();
    }
}
