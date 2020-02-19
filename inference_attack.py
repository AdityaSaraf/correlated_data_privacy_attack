import numpy as np
import matplotlib.pyplot as plt
import hmms

def symmetric_hmm(theta, rho):
    transitions = np.array([[1-theta,theta],
                            [theta,1-theta]])
    emissions = np.array([[1-rho/2,rho/2]
                         ,[rho/2,1-rho/2]])
    pi = [0.5, 0.5]
    return hmms.DtHMM(transitions, emissions, pi)

def eps_diff_privacy_noise(eps):
    return 2/(np.exp(eps)+1)


def emissions(latent_states, emission_probs):
    emissions = np.empty(len(latent_states),dtype=int)
    for i, state in enumerate(latent_states):
        emissions[i] =  np.random.choice(emission_probs.shape[1], 1, p = emission_probs[state,:].flatten())
    return emissions

def dp_attack(observations, attack_state):
    return [obs[attack_state] for obs in observations]

# Find the most likely last state by computing forwards probabilities
# Todo: extend with forwards-backwards algorithm to arbitrary positions
def correlation_attack(observations, attack_state, model):
    return [np.argmax(model.forward(obs)[-1]) for obs in observations]

# Find the most likely last state via the Viterbi algorithm
def correlation_attack_alt(observations, attack_state, model):
    return [model.viterbi(obs)[1][-1] for obs in observations]

if __name__ == "__main__":
    eps = 0.5
    print ('eps:', eps)
    noise = eps_diff_privacy_noise(eps)

    # The number of different hidden state models to test
    num_hidden_states = 100
    # The number of generated observations for each hidden state model
    num_observations = 1000
    # The length of the Markov chain
    seq_len = 30
    print('noise:', noise)
    print('Probability of not flipping state', 1-noise/2)
    model = symmetric_hmm(0.15, noise)
    latents, _ = model.generate_data((num_hidden_states, seq_len))
    attack_state = seq_len - 1

    dp_correct = 0
    corr_correct = 0
    for latent in latents:
        correct = latent[attack_state]
        observations = []
        for i in range(num_observations):
            observations.append(emissions(latent, model.b))
        dp_guesses = dp_attack(observations, attack_state)
        dp_correct += dp_guesses.count(correct)        
        corr_guesses = correlation_attack(observations, attack_state, model)
        corr_correct += corr_guesses.count(correct)
        print('Latent:', latent)
        # DP attacker's success probability should be close to 1 - noise/2
        print('DP attacker success probability:', dp_guesses.count(correct)/num_observations)
        print('Correlation attacker success probability:', corr_guesses.count(correct)/num_observations)

    dp_prob = dp_correct/(num_observations*num_hidden_states)
    print('Final DP attacker success probability:', dp_prob)
    print('Estimated eps: ', np.log(dp_prob/(1-dp_prob)))
  
    corr_prob = corr_correct/(num_observations*num_hidden_states)
    print('Final Correlation attacker success probability:', corr_prob)
    print('Estimated eps: ', np.log(corr_prob/(1-corr_prob)))

    # Graphs the first observation
    # plt.rcParams['figure.figsize'] = [20,20]
    # hmms.plot_hmm(latent, observed)