package neural_plswork.math;

/**
 * Much of the math to be done in neural networks involves Matrix operations.
 * This interface is used to add a layer of abstraction to matrices, allowing all elements of this interface
 * to be used within Matrices by defining a Multiply and Add instruction for them
 */

public interface MatrixElement<T> {

    /**
     * Function to multiply 2 objects. May be implemented in different ways per implemented class
     * @param other Other object to be multiplied with
     * @return Object result of multiplication
     */
    public T multiply(T other);

    /**
     * Function to add 2 objects. May be implemented in different ways per implemented class
     * @param other Other object to be add with
     * @return Object result of addition
     */
    public T add(T other);
}