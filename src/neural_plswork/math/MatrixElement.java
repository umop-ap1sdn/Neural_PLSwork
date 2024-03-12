package neural_plswork.math;

public interface MatrixElement<T> {
    public T multiply(T other);
    public T add(T other);
}