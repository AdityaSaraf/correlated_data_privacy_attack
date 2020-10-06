import numpy as np
import matplotlib.pyplot as plt
import hmms
import sys
import math

def symmetric_hmm(theta, rho):
    transitions = np.array([[1-theta,theta],
                            [theta,1-theta]])
    emissions = np.array([[1-rho,rho]
                         ,[rho,1-rho]])
    pi = [0.5, 0.5]
    return hmms.DtHMM(transitions, emissions, pi)

def diff_privacy_noise(eps):
    return 1/(np.exp(eps)+1)

def pf_privacy_noise(theta, eps):
    sqrt_term = math.sqrt(theta**2 * np.exp(eps) * (4+theta * (theta * np.exp(eps) - 4)))
    numer = 4 + theta * (theta * np.exp(eps) - 2) - sqrt_term
    denom = 8 + 2*theta*(theta * np.exp(eps) + theta - 4)
    noise = numer/denom
    assert noise <= 1 and noise >= 0
    return noise


def emissions(latent_states, emission_probs):
    emissions = np.empty(len(latent_states),dtype=int)
    for i, state in enumerate(latent_states):
        emissions[i] =  np.random.choice(emission_probs.shape[1], 1, p = emission_probs[state,:].flatten())
    return emissions

def dp_attack(observations, attack_state):
    return [obs[attack_state] for obs in observations]

# Find the most likely value of the state at index "attack_state"
def correlation_attack(observations, attack_state, model):
    return [np.argmax(model.states_confidence(obs)[attack_state]) for obs in observations]

# Find the most likely last state via the Viterbi algorithm
def correlation_attack_alt(observations, attack_state, model):
    return [model.viterbi(obs)[1][-1] for obs in observations]

if __name__ == "__main__":
    eps = 0.5
    # theta = 0.2
    print('eps:', eps, file=sys.stderr)
    dp_noise = diff_privacy_noise(eps)
    print('noise:', dp_noise, file=sys.stderr)
    print('Probability of not flipping state', 1-dp_noise, file=sys.stderr)
    
    # pf_noise = pf_privacy_noise(theta, eps)
    # print('pf_noise', pf_noise, file=sys.stderr)
    
    # The number of different hidden state models to test
    num_hidden_states = 100
    # The number of generated observations for each hidden state model
    num_observations = 2000
    # The length of the Markov chain
    seq_len = 30

    print('theta,suc_DP,suc_BDP,eps_DP,eps_BDP')
    for theta in np.linspace(0.02, 0.5, 11):
    # for theta in np.linspace(0.38, 0.5, 4):
        print('pf_noise:', pf_privacy_noise(theta, eps), 'at theta:', theta, file=sys.stderr)
        model = symmetric_hmm(theta, dp_noise)
        latents, _ = model.generate_data((num_hidden_states, seq_len))
        # latents = [np.full((100), 0, dtype=int)]
        attack_state = seq_len//2

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
            # print('Latent:', latent, file=sys.stderr)
            # DP attacker's success probability should be close to 1 - noise/2
            # print('DP attacker success probability:', dp_guesses.count(correct)/num_observations, file=sys.stderr)
            # print('Correlation attacker success probability:', corr_guesses.count(correct)/num_observations, file=sys.stderr)
        
        # Logging information
        print('theta', theta, file=sys.stderr)

        dp_prob = dp_correct/(num_observations*num_hidden_states)
        print('Final DP attacker success probability:', dp_prob, file=sys.stderr)
        print('Estimated eps: ', np.log(dp_prob/(1-dp_prob)), file=sys.stderr)
    
        corr_prob = corr_correct/(num_observations*num_hidden_states)
        print('Final Correlation attacker success probability:', corr_prob, file=sys.stderr)
        print('Estimated eps: ', np.log(corr_prob/(1-corr_prob)), file=sys.stderr)
        print('', file=sys.stderr)
        print('', file=sys.stderr)
        # Output
        print(theta, dp_prob, corr_prob, np.log(dp_prob/(1-dp_prob)), np.log(corr_prob/(1-corr_prob)), sep=',')

    # Graphs the first observation
    # plt.rcParams['figure.figsize'] = [20,20]
    # hmms.plot_hmm(latent, observed)