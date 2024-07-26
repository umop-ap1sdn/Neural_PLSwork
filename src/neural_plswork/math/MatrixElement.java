package neural_plswork.math;

/**
 * Much of the math to be done in neural networks involves Matrix operations.
 * This interface is used to add a layer of abstraction to matrices, allowing all elements of this interface
 * to be used within Matrices by defining a Multiply and Add instruction for them
 */

public interface MatrixElement {
    /**
     * Function to multiply 2 objects. May be implemented in different ways per implemented class
     * @param other Other object to be multiplied with
     * @return Object result of multiplication
     */
    public MatrixElement multiply(MatrixElement other);

    /**
     * Function to add 2 objects. May be implemented in different ways per implemented class
     * @param other Other object to be add with
     * @return Object result of addition
     */
    public MatrixElement add(MatrixElement other);

    /**
     * Function to create a deep copied element to ensure memory safety
     * @return Copy of the matrix element
     */
    public MatrixElement copy();

    /**
     * Function to confirm if 2 elements are compatible to be added
     * @param other Other object to be compared to
     * @return True if the 2 objects are compatible to be added
     */
    public boolean addable(MatrixElement other);

    /**
     * Function to confirm if 2 elements are compatible to be multiplied
     * @param other Other object to be compared to
     * @return True if the 2 objects are compatible to be multiplied
     */
    public boolean multipliable(MatrixElement other);

}