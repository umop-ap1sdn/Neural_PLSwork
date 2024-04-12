package neural_plswork.dataframe;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.HashMap;

public class DataFrame {
    private final String[] labels;
    private final HashMap<String, Integer> labelMap;
    private final Column<?>[] data;

    protected final int rows;
    protected final int columns;

    public DataFrame(String[] labels, Column<?>... data) {
        if(labels.length != data.length) throw new DataFrameException("Labels length must be equal to number of columns");
        this.labels = labels;
        this.data = data;

        if(data.length > 0) this.rows = data[0].size();
        else this.rows = 0;
        this.columns = labels.length;

        labelMap = new HashMap<>();
        for(int i = 0; i < labels.length; i++) {
            if(labelMap.containsKey(labels[i])) throw new DataFrameException("Cannot have identical labels in a data frame");
            labelMap.put(labels[i], i);
        }

        for(Column<?> c: data) {
            if(c.size() != rows) throw new DataFrameException("All columns must have the same length");
        }
    }

    public DataFrame loc(String... labels) {
        Column<?>[] located = new Column[labels.length];
        
        for(int i = 0; i < labels.length; i++) {
            if(!labelMap.containsKey(labels[i])) return null;
            located[i] = data[labelMap.get(labels[i])];
        }

        return new DataFrame(labels, located);
    }

    public DataFrame iloc(int... indices) {
        String[] labels = new String[indices.length];
        for(int i = 0; i < indices.length; i++) {
            labels[i] = this.labels[indices[i]];
        }

        return loc(labels);
    }

    public <T> void fillEmpty(String label, T value) {
        if(!labelMap.containsKey(label)) throw new DataFrameException("Input label not found");
        fillEmpty(labelMap.get(label), value);
    }

    @SuppressWarnings("unchecked")
    public <T> void fillEmpty(int column, T value) {
        try {
            ((Column<T>)data[column]).fillEmpty(value);
        } catch (ClassCastException e) {
            throw new DataFrameException("Input types do not match");
        }
    }

    public DataFrame appendColumn(String label, Column<?> column) {
        if(column.size() != rows) throw new DataFrameException("Column attempting to be appended has a wrong length");

        String[] labels = new String[columns + 1];
        Column<?>[] data = new Column[columns + 1];
        
        for(int i = 0; i < columns; i++) {
            labels[i] = this.labels[i];
            data[i] = this.data[i];
        }

        labels[columns] = label;
        data[columns] = column;

        return new DataFrame(labels, data);
    }

    public DataFrame append(Column<?>[] other) {
        if(other.length != columns) throw new DataFrameException("Appended array must have the same size");
        int otherRows = 0;
        if(other.length > 0) otherRows = other[0].size();
        for(Column<?> c: other) {
            if(c.size() != otherRows) throw new DataFrameException("Appended array must be rectangular");
        }

        Column<?>[] data = new Column[columns];
        for(int i = 0; i < columns; i++) {
            if(this.data[i].getClass() != other[i].getClass()) throw new DataFrameException("Columns must be of the same type");
            data[i] = this.data[i].append(other[i]);
            if(data[i] == null) throw new DataFrameException("Column types are incompatible");
        }

        return new DataFrame(labels, data);
    }

    public DataFrame append(DataFrame other) {
        if(columns != other.columns) throw new DataFrameException("DataFrames must have the same columns");
        for(int i = 0; i < labels.length; i++) {
            if(!labels[i].equals(other.labels[i])) throw new DataFrameException("DataFrames must have equivalent labels");
        }

        return append(other.data);
    }

    public DataFrame appendColumn(DataFrame other) {
        DataFrame appended = this;
        for(int i = 0; i < other.columns; i++) {
            appended = appended.appendColumn(other.labels[i], other.data[i]);
        }

        return appended;
    }

    public DataFrame slice(int start, int end) {
        Column<?>[] data = new Column<?>[columns];
        for(int i = 0; i < columns; i++) {
            data[i] = this.data[i].slice(start, end);
        }

        return new DataFrame(labels, data);
    }

    public DataFrame graduate() {
        Column<?>[] data = new Column[this.data.length];
        for(int i = 0; i < data.length; i++) {
            if(this.data[i].data instanceof String[]) data[i] = this.data[i].toStringColumn();
            else if(this.data[i].data instanceof Double[]) data[i] = this.data[i].toDoubleColumn();
            else if(this.data[i].data instanceof Integer[]) data[i] = this.data[i].toIntegerColumn();
            else if(this.data[i].data instanceof Boolean[]) data[i] = this.data[i].toBooleanColumn();
            else data[i] = this.data[i];
            
        }

        return new DataFrame(labels, data);
    }

    public DataFrame numify() {
        DoubleColumn[] data = new DoubleColumn[columns];
        for(int i = 0; i < columns; i++) {
            data[i] = this.data[i].numify().toDoubleColumn();
        }

        return new DataFrame(labels, data);
    }

    public boolean to_csv(String path, String name, boolean overwrite) {
        File file = new File(path);
        if(!file.exists()) file.mkdirs();

        int extPoint = name.indexOf(".");
        String fileName = name;
        String extension = ".csv";
        if(extPoint != -1) {
            fileName = name.substring(0, extPoint);
            extension = name.substring(extPoint);
        }

        File writeTo;

        if(!overwrite) {
            int id = 0;
            File check;
            do {
                String trueName = fileName + id + extension;
                check = new File(path + File.separator + trueName);
                id++;
            } while (check.exists());

            writeTo = check;
        } else {
            writeTo = new File(path + File.separator + fileName + extension);
        }

        try {
            FileWriter fw = new FileWriter(writeTo);
            
            String labelString = getRow(-1);
            fw.write(labelString + "\n");

            for(int i = 0; i < rows; i++) {
                fw.write(getRow(i) + "\n");
            }

            fw.close();

        } catch (IOException e) {
            return false;
        }

        return true;
    }

    public String getRow(int row) {
        if(columns == 0) return "";
        StringBuilder sb = new StringBuilder();
        
        if(row == -1) {
            sb.append(labels[0]);
            for(int i = 1; i < labels.length; i++) {
                sb.append("," + labels[i]);
            }

            return sb.toString();
        }

        sb.append(data[0].get(row).toString());

        for(int i = 1; i < columns; i++) {
            sb.append("," + data[i].get(row).toString());
        }

        return sb.toString();
    }

    public Column<?> getColumn(String label) {
        if(labelMap.containsKey(label)) return data[labelMap.get(label)];

        return null;
    }

    public String[] getLabels() {
        return labels;
    }

    public Column<?>[] getData() {
        return data;
    }

    @Override
    public String toString() {
        if(columns == 0) return "";
        StringBuilder sb = new StringBuilder();
        
        String labelString = getRow(-1);
        sb.append(labelString + "\n");

        for(int i = 0; i < rows; i++) {
            sb.append(getRow(i));
            if(i < rows - 1) sb.append("\n");
        }

        return sb.toString();
    }
}
