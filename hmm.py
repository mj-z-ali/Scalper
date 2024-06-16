import numpy as np
import time



def initial_pi_array(size: int, state_i: int) -> np.array:
    # Hidden state at index state_i will receive 
    # an initial state probability of 1 at time step 0. All other states
    # will receive a probability of 0.
    return np.array([[0] * state_i + [1] + [0] * (size-1-state_i)])

def initial_random_pi_array(size: int) -> np.array:

    # A random initial state probability distribution
    # for each  hidden state at timestep 0.
    random_array = np.random.rand(1, size)
    # Probabilities sum to 1.
    return random_array / np.sum(random_array)

def final_hidden_state_a_array(size: int, state_i: int) -> np.array:
    # Hidden state at index state_i will receive 
    # a probability of 1 for self-transition and 0 to all other states.
    return np.array([[0] * state_i + [1] + [0] * (size-1-state_i)])

def final_hidden_state_b_array(size: int, emission_k: int) -> np.array:
    # emission at index emission_k will receive 
    # a probability of 1 since final state can only emit it. 
    return np.array([[0] * emission_k + [1] + [0] * (size-1-emission_k)])

def initial_random_a_matrix(size: int, final_state_i: int) -> np.array:

    # Random transitional matrix without final state row.
    rand_a_mat_ex_final_state = np.random.rand(size-1,size)

    # Insert final state array.
    rand_a_mat = np.concatenate((
        rand_a_mat_ex_final_state[:final_state_i],
        final_hidden_state_a_array(size=size, state_i=final_state_i),
        rand_a_mat_ex_final_state[final_state_i:]
    ))

    # Normalize so all transisions sum to 1 for each state.
    return rand_a_mat / np.sum(rand_a_mat, axis=1).reshape(size, 1)

def initial_random_b_matrix(size_i: int, size_k: int, final_state_i: int, final_emission_k: int) -> np.array:

    # Random matrix excluding final state row and final emission column.
    rand_b_mat_ex_final = np.random.rand(size_i-1, size_k-1)

    # Normalize so all state to emission probabailities sum to 1.
    rand_b_mat_ex_final_norm = rand_b_mat_ex_final / np.sum(rand_b_mat_ex_final, axis=1).reshape(size_i-1, 1)

    # Insert 0 column at final emission column.
    rand_b_w_final_col = np.concatenate((
        rand_b_mat_ex_final_norm[:, :final_emission_k],
        np.zeros((size_i-1, 1)),
        rand_b_mat_ex_final_norm[:, final_emission_k:],
    ), axis=1)

    # Insert final state row.
    rand_b_mat = np.concatenate((
        rand_b_w_final_col[:final_state_i],
        final_hidden_state_b_array(size=size_k, emission_k=final_emission_k),
        rand_b_w_final_col[final_state_i:]
    ))

    return rand_b_mat

def alpha_0_array(pi_array: np.array, b_matrix: np.array, emission_k: int) -> np.array:
    
    # Initial alpha for each state at timestep 0.
    return pi_array * b_matrix[:, emission_k].T

def alpha_t_plus_one_array(alpha_t_array: np.array, a_matrix: np.array, 
                    b_matrix: np.array, emission_k: int) -> np.array:
    # Compute alpha of each hidden state for the next time step.
    return (alpha_t_array @ a_matrix) * b_matrix[:, emission_k].T

def beta_T_array(size: int, state_i: int) -> np.array:
    # Hidden state at index state_i will receive 
    # a beta of 1 at the final time step. All other states
    # will receive a beta of 0.
    return np.array([[0] * state_i + [1] + [0] * (size-1-state_i)])

def beta_t_minus_one_array(beta_t_array: np.array, a_matrix: np.array,
                        b_matrix: np.array, emission_k: int) -> np.array:

    # Product of state transitions and corresponding emission transitions.
    # Transposed so the following matrix multiplication of beta_t_array and 
    # a_b_product would result to Beta_i_(t) = Beta_0_(t+1) x a_i0 x b_0k + 
    # Beta_1_(t+1) x a_i1 x b_1k + ... + Beta_j_(t+1) x a_ij x b_jk for all 
    # states i and j.
    a_b_product = a_matrix.T * b_matrix[:, emission_k] 

    # Results in a beta array s.t. beta_t_array[i] is the beta value
    # for state i at time step t.
    return np.array([beta_t_array @ a_b_product])


def forward_matrix(alpha_t_matrix: np.array, a_matrix: np.array,
                b_matrix: np.array, observations: np.array,
                time_step: int) -> np.array:
    
    if time_step == observations.size - 1:

        return alpha_t_matrix
        
    # Compute alpha of each state at each time_step.
    # Result is a (num_of_hidden_states x observations_size) alpha_matrix,
    # Each alpha_matrix[i, t] is the the probability of being at state i at time step t
    # given the sequence up to that point.
    return forward_matrix(
            alpha_t_matrix=np.concatenate((
                            alpha_t_matrix, 
                            alpha_t_plus_one_array(alpha_t_array=alpha_t_matrix[-1], a_matrix=a_matrix, 
                                                b_matrix=b_matrix, emission_k=observations[:,time_step+1]))),
            a_matrix=a_matrix, b_matrix=b_matrix, observations=observations, time_step=time_step+1)

def forward_matrix_opt(alpha_t_matrix: np.array, a_matrix: np.array,
                b_matrix: np.array, observations: np.array) -> np.array:
    # forward matrix optimized for reduced call stack:

    # Compute alpha of each state at each time_step.
    # Result is a (num_of_hidden_states x observations_size) alpha_matrix,
    # Each alpha_matrix[i, t] is the the probability of being at state i at time step t
    # given the sequence up to that point.


    # The following branches reduces the call-stack by a factor of 3.
    if observations.size <= 1:

        return alpha_t_matrix
    
    alpha_t_matrix_1 = np.concatenate((
                            alpha_t_matrix, 
                            alpha_t_plus_one_array(alpha_t_array=alpha_t_matrix[-1], a_matrix=a_matrix, 
                                                b_matrix=b_matrix, emission_k=observations[:, 1])
                        ))

    if  observations.size == 2:

        return alpha_t_matrix_1
    
    alpha_t_matrix_2 = np.concatenate((
                            alpha_t_matrix_1, 
                            alpha_t_plus_one_array(alpha_t_array=alpha_t_matrix_1[-1], a_matrix=a_matrix, 
                                                b_matrix=b_matrix, emission_k=observations[:, 2])
                        ))
    
    if observations.size == 3:

        return alpha_t_matrix_2
    

    alpha_t_matrix_3 = np.concatenate((
                            alpha_t_matrix_2, 
                            alpha_t_plus_one_array(alpha_t_array=alpha_t_matrix_2[-1], a_matrix=a_matrix, 
                                                b_matrix=b_matrix, emission_k=observations[:, 3])
                        ))
        
    
    return forward_matrix_opt(
            alpha_t_matrix=alpha_t_matrix_3,
            a_matrix=a_matrix, b_matrix=b_matrix, observations=observations[:, 3:])
    
def backward_matrix(beta_t_matrix: np.array, a_matrix: np.array, b_matrix: np.array, observations: np.array, time_step: int) -> np.array:

    if time_step == 0:

        return beta_t_matrix
    
    # Compute beta of each state at each time_step.
    # Result is a (num_of_hidden_states x observations_size) beta_matrix,
    # Each beta_matrix[i, t] is the probability of ending the remaining sequence
    # given that in state i at timestep t.
    return backward_matrix(
            beta_t_matrix=np.concatenate((beta_t_minus_one_array(
                                            beta_t_array=beta_t_matrix[0], a_matrix=a_matrix, 
                                            b_matrix=b_matrix, emission_k=observations[:, time_step]), 
                                        beta_t_matrix)), 
            a_matrix=a_matrix, b_matrix=b_matrix, observations=observations, time_step=time_step-1)


def backward_matrix_opt(beta_t_matrix: np.array, a_matrix: np.array, b_matrix: np.array, observations: np.array) -> np.array:
    # Backward algorithm but optimized for reduced call-stack size.

    # Compute beta of each state at each time_step.
    # Result is a (num_of_hidden_states x observations_size) beta_matrix,
    # Each beta_matrix[i, t] is the probability of ending the remaining sequence
    # given that in state i at timestep t.


    # The following branches reduces the call-stack by a factor of 3.
    if observations.size <= 1:

        return beta_t_matrix
    
    beta_t_matrix_1 = np.concatenate((beta_t_minus_one_array(
                                            beta_t_array=beta_t_matrix[0], a_matrix=a_matrix, 
                                            b_matrix=b_matrix, emission_k=observations[:, -1]), 
                                        beta_t_matrix))
    
    if observations.size == 2:

        return beta_t_matrix_1
    

    beta_t_matrix_2 = np.concatenate((beta_t_minus_one_array(
                                            beta_t_array=beta_t_matrix_1[0], a_matrix=a_matrix, 
                                            b_matrix=b_matrix, emission_k=observations[:, -2]), 
                                        beta_t_matrix_1))
    
    if observations.size == 3:

        return beta_t_matrix_2
    

    beta_t_matrix_3 = np.concatenate((beta_t_minus_one_array(
                                            beta_t_array=beta_t_matrix_2[0], a_matrix=a_matrix, 
                                            b_matrix=b_matrix, emission_k=observations[:, -3]), 
                                        beta_t_matrix_2))
    
    if observations.size == 4:

        return beta_t_matrix_3
    
    beta_t_matrix_4 = np.concatenate((beta_t_minus_one_array(
                                            beta_t_array=beta_t_matrix_3[0], a_matrix=a_matrix, 
                                            b_matrix=b_matrix, emission_k=observations[:, -4]), 
                                        beta_t_matrix_3))

    
    return backward_matrix_opt(
            beta_t_matrix=beta_t_matrix_4, 
            a_matrix=a_matrix, b_matrix=b_matrix, observations=observations[:, :-4])

def gamma_matrix(alpha_matrix: np.array, beta_matrix: np.array) -> np.array:

    alpha_beta_product_matrix = alpha_matrix * beta_matrix
    # Results to a (num_of_hidden_states x num_of_observations) matrix.
    # Each gamma[t,i] is the probability of being at state i at time t given
    # the total sequence Y and model parameters.
    
    return alpha_beta_product_matrix / np.sum(alpha_beta_product_matrix, axis=1).reshape(alpha_beta_product_matrix.shape[0], 1)

def xi_t_array(alpha_t_array: np.array, beta_t_plus_one_array: np.array, a_matrix: np.array, b_matrix: np.array, emission_k: int) -> np.array:

    # Results in a (num_of_hidden_states, num_of_hidden_states) matrix.
    # All Xi[i,j] is the xi value of state  i to j.
    return alpha_t_array.T * a_matrix * b_matrix[:, emission_k].T * beta_t_plus_one_array

def xi_matrix(xi_mat: np.array, alpha_matrix: np.array, beta_matrix: np.array, a_matrix: np.array, b_matrix: np.array, observations: np.array, time_step: int) -> np.array:

    # Result is a (observations_size - 1, num_of_hidden_states, num_of_hidden_states) matrix.
    # Each Xi[t,i,j] is the probability of being in state i and j at times t and t+1 respectively
    # given the total observed sequence Y and parameters theta. Denominators of gamma[i,t] and xi[t,i,j]
    # are the same, that is, the probability of making observation Y given the model parameters theta.
    if time_step == observations.size - 1:
        return xi_mat / np.sum(xi_mat, axis=(1,2)).reshape(time_step, 1, 1)
    
    # Xi[i,j] is the xi value of state  i to j for the current timestep.
    xi_t_layer = xi_t_array(alpha_t_array=np.array([alpha_matrix[time_step]]), beta_t_plus_one_array=np.array([beta_matrix[time_step+1]]), a_matrix=a_matrix, b_matrix=b_matrix, emission_k=observations[:, time_step+1])


    xi_zero_to_t_matrix = np.concatenate((
        xi_mat,
        np.array([xi_t_layer]),
    )) 

    return xi_matrix(xi_mat=xi_zero_to_t_matrix, alpha_matrix=alpha_matrix, beta_matrix=beta_matrix, a_matrix=a_matrix, b_matrix=b_matrix, observations=observations, time_step=time_step+1)

def learn_pi_array(g_matrix: np.array) -> np.array:

    return g_matrix[0].reshape(1, g_matrix.shape[1])

def learn_a_matrix(xi_mat: np.array, g_matrix: np.array, final_state_i: int) -> np.array:

    # Transpose in order to do easy summations.
    # We have xi[i,t,j].
    # Sum each state pair transition with respect to time t.
    # Results in a (num_of_hidden_states, num_of_hidden_states) matrix s.t.
    # a_matrix_numerator[i,j] is the sum of xi values from state i to j across all timesteps.
    a_matrix_numerator = np.sum(np.transpose(xi_mat, (1,0,2)), axis=1)

    # Sum each gamma[t,i] with respect to time t, excluding the final timestep since xi matrix
    # excludes it. The result after transposing is a (num_of_hidden_states x 1) matrix.
    a_matrix_denominator_with_zero = np.sum(g_matrix[:-1], axis=0).reshape(g_matrix.shape[1], 1)

    # Fix divide-by-zero error due to final hidden state having 0 gammas at all timesteps but the last.
    a_matrix_denominator = np.where(a_matrix_denominator_with_zero==0, 1, a_matrix_denominator_with_zero)

    learned_a_matrix = a_matrix_numerator / a_matrix_denominator

    # Replace final hidden state row where its transitional probability to itself is 1
    # and 0 everywhere else.
    return np.concatenate((learned_a_matrix[:final_state_i], 
                                                        final_hidden_state_a_array(size=learned_a_matrix.shape[0],state_i=final_state_i), 
                                                        learned_a_matrix[final_state_i+1:]
                                                    ))


def observation_occurence_matrix(observations:np.array, number_of_emissions: int) -> np.array:

    emissions = np.arange(number_of_emissions).reshape(number_of_emissions,1)

    # A (number_of_emissions, observations_size) one-hot encoded matrix s.t.
    # observation_occurence_matrix[i,j] denotes if emission_i occurs in observation_j.
    return  (emissions == observations).astype(int)

def learn_b_matrix(g_matrix: np.array, observations: np.array, number_of_emissions: int) -> np.array:

    # Reshape observation_occurence_matrix to a (number_of_emissions, observations_size,1) matrix.
    observation_occurence_matrix_reshape = observation_occurence_matrix(observations=observations, number_of_emissions=number_of_emissions)[:,:,np.newaxis]

    # Results in a (number_of_emissions x observations_size x num_of_hidden_states) matrix s.t.
    # gamma_occurence[k,t,i] is the gamma value of state i at timestep t if emission k occurs in
    # in that timestep. Otherwise, it is zero.
    gamma_occurence = g_matrix  * observation_occurence_matrix_reshape

    # Sum each gamma[t,i] with respect to time for each emission.
    # Gammas where emission k does not occur at that timestep are 0.
    # Results in a (number_of_emissions x num_of_hidden_states) matrix.
    b_matrix_numerator = np.sum(gamma_occurence, axis=1)

    # Sum each gamma[t,i] with respect to time t.
    # The result after transposing is a (num_of_hidden_states x 1) matrix.
    b_matrix_denominator = np.sum(g_matrix, axis=0).reshape(g_matrix.shape[1], 1)

    # Results in a (num_of_hidden_states x num_of_emissions) matrix.
    # That is, the updated b matrix.
    return b_matrix_numerator.T / b_matrix_denominator


def baum_welch(pi_array: np.array, a_matrix: np.array, b_matrix: np.array, observations: np.array, final_state_i: int) -> np.array:
    
    alpha_matrix = forward_matrix_opt(alpha_t_matrix=alpha_0_array(pi_array=pi_array, b_matrix=b_matrix, emission_k=observations[:,0]),
                            a_matrix=a_matrix, b_matrix=b_matrix, observations=observations)

    beta_matrix = backward_matrix_opt(beta_t_matrix=beta_T_array(size=a_matrix.shape[0], state_i=final_state_i), 
                            a_matrix=a_matrix, b_matrix=b_matrix, observations=observations)

    # Result is a (num_of_timestep, num_of_hidden_states, num_of_hidden_states) matrix s.t.,
    # every gamma_mat[t,i,j] is the gamma of hidden states i and j at timestep t.
    g_matrix = gamma_matrix(alpha_matrix=alpha_matrix, beta_matrix=beta_matrix)

    init_xi_mat = np.empty((0, a_matrix.shape[0], a_matrix.shape[1]))
    xi_mat = xi_matrix(xi_mat=init_xi_mat, alpha_matrix=alpha_matrix, beta_matrix=beta_matrix, a_matrix=a_matrix, b_matrix=b_matrix, observations=observations, time_step=0)

    learned_pi_array = learn_pi_array(g_matrix=g_matrix)

    learned_a_matrix = learn_a_matrix(xi_mat=xi_mat, g_matrix=g_matrix, final_state_i=final_state_i)

    learned_b_matrix = learn_b_matrix(g_matrix=g_matrix, observations=observations, number_of_emissions=b_matrix.shape[1])

    return learned_pi_array, learned_a_matrix, learned_b_matrix

def train_until_convergence(pi_array: np.array, a_matrix: np.array, b_matrix: np.array, multiple_observations: np.array, final_state_i: int, thresh: float, iteration: int, max_iterations: int) -> np.array:
    # Learned pi, learned a,  and learned b are the sum of pi, a, and b matrices of all
    # observations - that resulted from their respective Baum-Welch step - divided by the 
    # number of total observations.
    # This process is repeated until desired level of convergence is met.
    learned_pi_array, learned_a_matrix, learned_b_matrix = train_each_observation(pi_array=pi_array, a_matrix=a_matrix, b_matrix=b_matrix, learned_pi_array=np.zeros(pi_array.shape), learned_a_matrix=np.zeros(a_matrix.shape), learned_b_matrix=np.zeros(b_matrix.shape), multiple_observations=multiple_observations, final_state_i=final_state_i, R=multiple_observations.shape[0])

    if (np.linalg.norm(pi_array-learned_pi_array) < thresh and np.linalg.norm(a_matrix-learned_a_matrix) < thresh and np.linalg.norm(b_matrix-learned_b_matrix) < thresh) or (iteration == max_iterations):

        return learned_pi_array, learned_a_matrix, learned_b_matrix, iteration
    
    return train_until_convergence(pi_array=learned_pi_array, a_matrix=learned_a_matrix, b_matrix=learned_b_matrix, multiple_observations=multiple_observations, final_state_i=final_state_i, thresh=thresh, iteration=iteration+1, max_iterations=max_iterations)
    
def train_each_observation(pi_array: np.array, a_matrix: np.array, b_matrix: np.array, learned_pi_array: np.array, learned_a_matrix: np.array, learned_b_matrix: np.array, multiple_observations: np.array, final_state_i: int, R: int) -> np.array:
    
    # Compute Baum-Welch on each observation. Then, sum the learned pi, a, and b matrices
    # from each Baum-Welch for each observation and divide the final result with the number
    # of observations.

    if multiple_observations.shape[0] == 0:

        return learned_pi_array / R, learned_a_matrix / R, learned_b_matrix / R

    # The following branches reduce the stack size by a factor of 3:

    learned_pi_array_1, learned_a_matrix_1, learned_b_matrix_1 = baum_welch(
        pi_array=pi_array, a_matrix=a_matrix, b_matrix=b_matrix,
        observations=multiple_observations[0].reshape(1, multiple_observations.shape[1]), 
        final_state_i=final_state_i)
    
    if multiple_observations.shape[0] == 1:

        return (learned_pi_array+learned_pi_array_1) / R, \
               (learned_a_matrix+learned_a_matrix_1) / R, \
               (learned_b_matrix+learned_b_matrix_1) / R
    

    learned_pi_array_2, learned_a_matrix_2, learned_b_matrix_2 = baum_welch(
        pi_array=pi_array, a_matrix=a_matrix, b_matrix=b_matrix, 
        observations=multiple_observations[1].reshape(1, multiple_observations.shape[1]), 
        final_state_i=final_state_i)
    
    if multiple_observations.shape[0] == 2:

        return (learned_pi_array+learned_pi_array_1+learned_pi_array_2) / R, \
               (learned_a_matrix+learned_a_matrix_1+learned_a_matrix_2) / R, \
               (learned_b_matrix+learned_b_matrix_1+learned_b_matrix_2) / R



    learned_pi_array_3, learned_a_matrix_3, learned_b_matrix_3 = baum_welch(
    pi_array=pi_array, a_matrix=a_matrix, b_matrix=b_matrix, 
    observations=multiple_observations[2].reshape(1, multiple_observations.shape[1]), 
    final_state_i=final_state_i)

    return train_each_observation(
                    pi_array=pi_array, a_matrix=a_matrix, b_matrix=b_matrix,
                    learned_pi_array=learned_pi_array+learned_pi_array_1+learned_pi_array_2+learned_pi_array_3, 
                    learned_a_matrix=learned_a_matrix+learned_a_matrix_1+learned_a_matrix_2+learned_a_matrix_3,
                    learned_b_matrix=learned_b_matrix+learned_b_matrix_1+learned_b_matrix_2+learned_b_matrix_3, 
                    multiple_observations=multiple_observations[3:],
                    final_state_i=final_state_i, R=R)
    

def train(multiple_observations: np.array, number_of_hidden_states: int, number_of_emissions: int, final_state_i: int, final_emission_k: int, thresh: float, max_iterations: int) -> np.array:

    # Initial pi, a, and b matrices. Probabilities for final state and final emission are accounted for.
    pi_array = initial_random_pi_array(size=number_of_hidden_states)
    a_matrix = initial_random_a_matrix(size=number_of_hidden_states, final_state_i=final_state_i)
    b_matrix = initial_random_b_matrix(size_i=number_of_hidden_states, size_k=number_of_emissions, 
                                    final_state_i=final_state_i, final_emission_k=final_emission_k)
    

    return train_until_convergence(pi_array=pi_array, a_matrix=a_matrix, b_matrix=b_matrix, multiple_observations=multiple_observations, final_state_i=final_state_i, thresh=thresh, iteration=0, max_iterations=max_iterations)


a_matrix = np.array([[1,0,0,0], [0.2,0.3,0.1,0.4], [0.2,0.5,0.2,0.1], [0.7,0.1,0.1,0.1]])
b_matrix = np.array([[1,0,0,0,0], [0,0.3,0.4,0.1,0.2], [0,0.1,0.1,0.7,0.1], [0,0.5,0.2,0.1,0.2]])
multiple_observations  = np.array([[4,1,3,2,2,3,4,1,3,2,3,2,3,2,0]])

print(f"a_matrix shape {a_matrix.shape}")
print(f"a_matrix \n {a_matrix}")
print(np.allclose(a_matrix.sum(axis=1), 1))

print(f"b_matrix shape {b_matrix.shape}")
print(f"b_matrix \n {b_matrix}")
print(np.allclose(b_matrix.sum(axis=1), 1))


print(f"observations shape {multiple_observations.shape}")
print(f"observations \n {multiple_observations}")


learned_pi_array, learned_a_matrix, learned_b_matrix, iterations = train(multiple_observations=multiple_observations, number_of_hidden_states=4, number_of_emissions=5, final_state_i=0, final_emission_k=0, thresh=0.00001, max_iterations=100)
print(f"learned pi array shape {learned_pi_array.shape}")
print(f"learned pi array \n {learned_pi_array}")
print(np.allclose(learned_pi_array.sum(axis=1), 1))

print(f"learned a_matrix shape {learned_a_matrix.shape}")
print(f"learned a_matrix \n {learned_a_matrix}")
print(np.allclose(learned_a_matrix.sum(axis=1), 1))

print(f"learned b_matrix shape {learned_b_matrix.shape}")
print(f"learned b_matrix \n {learned_b_matrix}")
print(np.allclose(learned_b_matrix.sum(axis=1), 1))

print(f"Baum-Welch iterations {iterations}")


data = np.append(np.random.randint(1,5, (1, 99)),0).reshape(1,100)


s_time_hmm = time.time()
pi, a, b, i = train(multiple_observations=data, number_of_hidden_states=4, number_of_emissions=5, final_state_i=0, final_emission_k=0, thresh=0.00001, max_iterations=1000)
e_time_hmm = time.time()



print("="*5, "HMM", "="*5)
print(f"pi \n {pi}")
print(np.allclose(pi.sum(axis=1), 1))
print(f"a matrix \n {a}")
print(np.allclose(a.sum(axis=1), 1))
print(f"b matrix \n {b}")
print(np.allclose(b.sum(axis=1), 1))
print(f"iterations {i}")
print(f"time elapsed {e_time_hmm - s_time_hmm}")
print("="*10)

