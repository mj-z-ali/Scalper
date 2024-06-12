import numpy as np

def alpha_0_array(size: int, state_i: int) -> np.array:
    # Hidden state at index state_i will receive 
    # an alpha of 1 at time step 0. All other states
    # will receive an alpha of 0.
    return np.array([[0] * state_i + [1] + [0] * (size-1-state_i)])


def alpha_t_plus_one_array(alpha_t_array: np.array, a_matrix: np.array, 
                    b_matrix: np.array, emission_k: int) -> np.array:
    # Compute alpha of each hidden state for the next time step.
    return (alpha_t_array @ a_matrix) * b_matrix[:, emission_k].T


def forward_matrix(alpha_t_matrix: np.array, a_matrix: np.array,
                b_matrix: np.array, observations: np.array,
                time_step: int) -> np.array:
    
    if time_step == observations.size:

        return alpha_t_matrix
    
    # Compute alpha of each state at each time_step.
    # Result is a (num_of_hidden_states x observations_size + 1) alpha_matrix,
    # plus 1 since time step 0 is included. 
    # Each alpha_matrix[i, t] is the alpha of state i at time step t.
    return forward_matrix(
            alpha_t_matrix=np.concatenate((
                            alpha_t_matrix, 
                            alpha_t_plus_one_array(alpha_t_array=alpha_t_matrix[-1], a_matrix=a_matrix, 
                                                b_matrix=b_matrix, emission_k=observations[:,time_step]))),
            a_matrix=a_matrix, b_matrix=b_matrix, observations=observations, time_step=time_step+1)
    

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

def backward_matrix(beta_t_matrix: np.array, a_matrix: np.array, b_matrix: np.array, observations: np.array, time_step: int) -> np.array:

    if time_step == -1:

        return beta_t_matrix
    
    # Compute beta of each state at each time_step.
    # Result is a (num_of_hidden_states x observations_size + 1) beta_matrix,
    # plus 1 since time step 0 is included. 
    # Each beta_matrix[i, t] is the beta of state i at time step t.
    return backward_matrix(
            beta_t_matrix=np.concatenate((beta_t_minus_one_array(
                                            beta_t_array=beta_t_matrix[0], a_matrix=a_matrix, 
                                            b_matrix=b_matrix, emission_k=observations[:, time_step]), 
                                        beta_t_matrix)), 
            a_matrix=a_matrix, b_matrix=b_matrix, observations=observations, time_step=time_step-1)


def gamma_t_matrix(alpha_t_minus_one_array: np.array, beta_t_array: np.array, a_matrix: np.array, b_matrix: np.array, emission_k: int) -> np.array:

    # The gamma for every state pair.
    # Results in a (num_of_hidden_states, num_of_hidden_states) matrix.
    return alpha_t_minus_one_array.T * a_matrix * b_matrix[:, emission_k].T * beta_t_array

def gamma_matrix(alpha_matrix: np.array, beta_matrix: np.array, a_matrix: np.array, b_matrix: np.array, observations: np.array, time_step: int) -> np.array:

    if time_step == observations.size:
        return np.empty((0, a_matrix.shape[0], a_matrix.shape[1]))
    
    
    gamma_t_mat = gamma_t_matrix(alpha_t_minus_one_array=np.array([alpha_matrix[time_step]]), beta_t_array=np.array([beta_matrix[time_step+1]]), a_matrix=a_matrix, b_matrix=b_matrix, emission_k=observations[:, time_step])
    
    gamma_t_to_T_mat = gamma_matrix(alpha_matrix=alpha_matrix, beta_matrix=beta_matrix, a_matrix=a_matrix, b_matrix=b_matrix, observations=observations, time_step=time_step+1)

    # Result is a (num_of_timestep, num_of_hidden_states, num_of_hidden_states) matrix s.t.,
    # every gamma_matrix[t,i,j] is the gamma of hidden states i and j at timestep t.
    gamma_mat = np.concatenate((
        np.array([gamma_t_mat]),
        gamma_t_to_T_mat
    ))

    print(np.sum(alpha_matrix[-1]))
    print(alpha_matrix)
    return gamma_mat / np.sum(alpha_matrix[-1])

def final_hidden_state_a_array(size: int, state_i: int) -> np.array:
    # Hidden state at index state_i will receive 
    # a probability of 1 for self-transition and 0 to all other states.
    return np.array([[0] * state_i + [1] + [0] * (size-1-state_i)])

def final_hidden_state_b_array(size: int, emission_k: int) -> np.array:
    # emission at index emission_k will receive 
    # a probability of 1 since final state can only emit it. 
    return np.array([[0] * emission_k + [1] + [0] * (size-1-emission_k)])

def gamma_denominator_array(g_matrix: np.array) -> np.array:
    
    g_matrix_transpose = np.transpose(g_matrix,(1,0,2))
    # Results in a (num_of_hidden_states) matrix s.t.
    # d[i] is the sum of all state gammas transitioned from
    # state i across all timesteps.
    g_denominator_array = np.array([np.sum(g_matrix_transpose, axis=(1,2))]).T

    # Fix divide-by-zero error due to final hidden state having 0 gammas at all timesteps.
    return np.where(g_denominator_array==0, 1, g_denominator_array)

def learned_a_numerator_matrix(g_matrix: np.array) -> np.array:

    # Transpose in order to do easy summations.
    # We have gamma_mat_transpose[i,t,j]
    gamma_mat_transpose = np.transpose(g_matrix, (1, 0, 2))

    # Sum each state pair transition with respect to time t.
    # Results in a (num_of_hidden_states, num_of_hidden_states) matrix s.t.
    # learned_a_numerator[i,j] is the sum of gammas from state i to j across all timesteps.
    return np.sum(gamma_mat_transpose, axis=1)


def learn_a_matrix(g_matrix: np.array, g_denominator_array: np.array, number_of_hidden_states: int, final_state_i: int) -> np.array:

    # Updated a_matrix but final state has 0 transitions due to an alpha of 0 in all timesteps.
    learned_a_matrix = learned_a_numerator_matrix(g_matrix=g_matrix) / g_denominator_array

    # Replace final state transition row.
    return np.concatenate((learned_a_matrix[:final_state_i], 
                                                        final_hidden_state_a_array(size=number_of_hidden_states,state_i=final_state_i), 
                                                        learned_a_matrix[final_state_i+1:]
                                                    ))


def observation_occurence_matrix(observations:np.array, number_of_emissions: int) -> np.array:

    emissions = np.array([np.arange(number_of_emissions)]).T

    # A (number_of_emissions, observations_size) one-hot encoded matrix s.t.
    # observation_occurence_matrix[i,j] denotes if emission_i occurs in observation_j.
    return  (emissions == observations).astype(int)

def learned_b_numerator_matrix(g_matrix: np.array, number_of_emissions: int) -> np.array:

    # Reshape observation_occurence_matrix to a (1,1,number_of_emissions, observations_size) matrix.
    observation_occurence_matrix_reshape = observation_occurence_matrix(observations=observations, number_of_emissions=number_of_emissions)[:,:,np.newaxis,np.newaxis]

    # Reshape gamma matrix to a (1, observations_size, num_of_hidden_states, num_of_hidden_states) matrix.
    gamma_matrix_reshape = g_matrix[np.newaxis,:,:,:]

    # A (number_of_emissions, num_of_hidden_states, timesteps, num_of_hidden_states) matrix.
    # For  obs_occur_gamma_product_matrix[k, i, t, j], if emission_k does not occur at timestep t, then
    # all gamma values are zero for those k,t pairs.
    # Matrix is transposed for easier summation.
    obs_occur_gamma_product_matrix = (observation_occurence_matrix_reshape * gamma_matrix_reshape).transpose(0,2,1,3)
    
    # Results to a (num_of_hidden_states, num_of_emissions) matrix[j,k] s.t
    # each value is the numerator for computing b_jk for hidden_state_j and emission_k.
    return np.sum(obs_occur_gamma_product_matrix, axis=(2,3)).T

def learn_b_matrix(g_matrix: np.array, g_denominator_array: np.array, number_of_emissions: int, final_state_i: int, final_emission_k: int) -> np.array:

    # Results in a (num_of_hidden_states, num_of_emissions) matrix s.t.
    # each m[j,k] is the updated b value for hidden state j and emission k.
    learned_b_matrix = learned_b_numerator_matrix(g_matrix=g_matrix, number_of_emissions=number_of_emissions) / g_denominator_array

    # Replace final state row to since it can only emit final emission.
    return np.concatenate((learned_b_matrix[:final_state_i], 
                            final_hidden_state_b_array(size=number_of_emissions, emission_k=final_emission_k), 
                            learned_b_matrix[final_state_i+1:]
                        ))

def baum_welch(a_matrix: np.array, b_matrix: np.array, observations: np.array, initial_state_i: int, final_state_i: int, final_emission_k: int, iterations: int, max_iterations: int) -> np.array:

    alpha_matrix = forward_matrix(alpha_t_matrix=alpha_0_array(size=a_matrix.shape[0], state_i=initial_state_i),
                            a_matrix=a_matrix, b_matrix=b_matrix, observations=observations, time_step=0)

    beta_matrix = backward_matrix(beta_t_matrix=beta_T_array(size=a_matrix.shape[0], state_i=final_state_i), 
                            a_matrix=a_matrix, b_matrix=b_matrix, observations=observations, time_step=observations.size-1)
    # Result is a (num_of_timestep, num_of_hidden_states, num_of_hidden_states) matrix s.t.,
    # every gamma_mat[t,i,j] is the gamma of hidden states i and j at timestep t.
    g_matrix = gamma_matrix(alpha_matrix=alpha_matrix, beta_matrix=beta_matrix, a_matrix=a_matrix, b_matrix=b_matrix, observations=observations, time_step=0)
    
    g_denominator_array = gamma_denominator_array(g_matrix=g_matrix)

    learned_a_matrix = learn_a_matrix(g_matrix=g_matrix, g_denominator_array=g_denominator_array, number_of_hidden_states=a_matrix.shape[0], final_state_i=final_state_i)

    learned_b_matrix = learn_b_matrix(g_matrix=g_matrix, g_denominator_array=g_denominator_array, number_of_emissions=b_matrix.shape[1], final_state_i=final_state_i, final_emission_k=final_emission_k)

    if (np.linalg.norm(a_matrix-learned_a_matrix) < .00001 and np.linalg.norm(b_matrix-learned_b_matrix) < .00001) or (iterations == max_iterations):

        return learned_a_matrix, learned_b_matrix, iterations+1
    
    return baum_welch(a_matrix=learned_a_matrix, b_matrix=learned_b_matrix, observations=observations, 
                    initial_state_i=initial_state_i, final_state_i=final_state_i, 
                    final_emission_k=final_emission_k, iterations=iterations+1, max_iterations=max_iterations)

a_matrix = np.array([[1,0,0,0], [0.2,0.3,0.1,0.4], [0.2,0.5,0.2,0.1], [0.7,0.1,0.1,0.1]])
b_matrix = np.array([[1,0,0,0,0], [0,0.3,0.4,0.1,0.2], [0,0.1,0.1,0.7,0.1], [0,0.5,0.2,0.1,0.2]])
observations  = np.array([[1,3,2,0]])

print(f"a_matrix shape {a_matrix.shape}")
print(f"a_matrix \n {a_matrix}")
print(np.allclose(a_matrix.sum(axis=1), 1))

print(f"b_matrix shape {b_matrix.shape}")
print(f"b_matrix \n {b_matrix}")
print(np.allclose(b_matrix.sum(axis=1), 1))


print(f"observations shape {observations.shape}")
print(f"observations {observations}")


learned_a_matrix, learned_b_matrix, iterations = baum_welch(a_matrix=a_matrix, b_matrix=b_matrix, observations=observations, initial_state_i=1, final_state_i=0, final_emission_k=0, iterations=0, max_iterations=10)

print(f"learned a_matrix shape {learned_a_matrix.shape}")
print(f"learned a_matrix \n {learned_a_matrix}")
print(np.allclose(learned_a_matrix.sum(axis=1), 1))

print(f"learned b_matrix shape {learned_b_matrix.shape}")
print(f"learned b_matrix \n {learned_b_matrix}")
print(np.allclose(learned_b_matrix.sum(axis=1), 1))

print(f"Baum-Welch iterations {iterations}")




