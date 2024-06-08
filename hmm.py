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


a_matrix = np.array([[1,0,0,0], [0.2,0.3,0.1,0.4], [0.2,0.5,0.2,0.1], [0.7,0.1,0.1,0.1]])
b_matrix = np.array([[1,0,0,0,0], [0,0.3,0.4,0.1,0.2], [0,0.1,0.1,0.7,0.1], [0,0.5,0.2,0.1,0.2]])
observations  = np.array([[1,3,2,0]])

alpha_matrix = forward_matrix(alpha_t_matrix=alpha_0_array(size=4, state_i=1),
                            a_matrix=a_matrix, b_matrix=b_matrix, observations=observations, time_step=0)

beta_matrix = backward_matrix(beta_t_matrix=beta_T_array(size=4, state_i=0), a_matrix=a_matrix, b_matrix=b_matrix, observations=observations, time_step=observations.size-1)
print(f"alpha matrix shape {alpha_matrix.shape}")
print(f"alpha matrix {alpha_matrix}")

print(f"beta matrix shape {beta_matrix.shape}")
print(f"beta matrix {beta_matrix}")
