package neural_plswork.dataframe;

import java.util.HashMap;

public class Column<T> {
    protected T[] data;
    
    public Column(T[] data) {
        this.data = data;
    }

    @SuppressWarnings("unchecked")
    public Column<T> slice(int start, int end) {
        T[] sliced = (T[]) new Object[end - start];
        for(int i = 0; i < sliced.length; i++) {
            sliced[i] = data[i + start];
        }

        return new Column<>(sliced);
    }

    @SuppressWarnings("unchecked")
    public Column<T> append(Column<T> other) {
        T[] appended = (T[]) new Object[data.length + other.data.length];
        for(int i = 0; i < appended.length; i++) {
            if(i < data.length) appended[i] = data[i];
            else appended[i] = other.data[i - data.length];
        }

        return new Column<>(appended);
    }

    @SuppressWarnings("unchecked")
    public Column<T> fillEmpty(T value) {
        T[] arr = (T[]) new Object[data.length];
        for(int i = 0; i < arr.length; i++) {
            if(data[i] == null) arr[i] = value;
            else arr[i] = data[i];
        }

        return new Column<>(data);
    }

    public T get(int index) {
        return data[index];
    }

    public int size() {
        return data.length;
    }

    public Column<Double> numify() {
        HashMap<T, Double> map = new HashMap<>();
        Double[] numbers = new Double[data.length];

        double count = 0.0;
        for(int i = 0; i < data.length; i++) {
            if(!map.containsKey(data[i])) map.put(data[i], count++);
            numbers[i] = map.get(data[i]);
        }

        return new Column<Double>(numbers);
    }

    public Column<Double> numify(HashMap<T, Double> mapPreset) {
        if(mapPreset == null) return numify();
        Double[] numbers = new Double[data.length];

        double count = mapPreset.size();
        for(int i = 0; i < data.length; i++) {
            if(!mapPreset.containsKey(data[i])) mapPreset.put(data[i], count++);
            numbers[i] = mapPreset.get(data[i]);
        }

        return new Column<Double>(numbers);
    }
}
