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

    public void addTransformation(Transformation transformation) {
        currentData = transformation.fit_transform(currentData);
        path.add(transformation);
        if(transformation instanceof Reversible) revPath.add(0, (Reversible) transformation);
    }

    public DoubleColumn[] transform(DoubleColumn[] input) {
        DoubleColumn[] data = input;
        for(Transformation t: path) data = t.transform(data);
        return data;
    }

    public DoubleColumn[] transform(NumericalDataFrame ndf) {
        return transform(ndf.getData());
    }

    public DoubleColumn[] reverse(DoubleColumn[] input) {
        DoubleColumn[] data = input;
        for(Reversible r: revPath) data = r.reverse(data);
        return data;
    }

    public DoubleColumn[] reverse(NumericalDataFrame ndf) {
        return reverse(ndf.getData());
    }

    public DoubleColumn[] getCurrent() {
        return currentData;
    }
}
