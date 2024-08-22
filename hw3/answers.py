r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""


# ==============
# Part 1 answers


def part1_rnn_hyperparams():
    hypers = dict(
        batch_size=128,
        seq_len=100,
        h_dim=256,
        n_layers=3,
        dropout=0.2,
        learn_rate=0.001,
        lr_sched_factor=0.1,
        lr_sched_patience=5,
    )
    # TODO: Set the hyperparameters to train the model.
    # ====== YOUR CODE: ======

    # ========================
    return hypers


def part1_generation_params():
    start_seq = "SCENE:"
    temperature = 0.01
    # TODO: Tweak the parameters to generate a literary masterpiece.
    # ====== YOUR CODE: ======

    # ========================
    return start_seq, temperature


part1_q1 = r"""
**Your answer:**
Training on all the text will be very expensive in run time and memory. Because  in this way our model will need to work
with a lot of tokens. But with shorter sequences, we can save run time and memory. In Addition, if we train on the
whole text, which is very large, the gradients can become unstable and we will have a problems of
vanishing or exploding gradients. But with short sequences, the gradients will stay stable.  
"""

part1_q2 = r"""
**Your answer:**
The model saves important information as context in the hidden states and moving them forward, and extends the model's
memory.
"""

part1_q3 = r"""
**Your answer:**
We don't shuffling the order of the batches because it helps the model to pay attention to the context, because we have
sequential data, shuffling would break the coherence of sentences and make it impossible for the model to learn. 
"""

part1_q4 = r"""
**Your answer:**
1.We lower the temperature to make the prediction of the model more deterministic and less random. 
  With lower temperature the probability distribution will be more sharp and the prediction will be more connected to the
  context. 
2.The probability distribution becomes more uniform, and sampling becomes more random and unpredictable.
  The model is more likely to choose lower-probability tokens ant the outputs will be more diverse but less coherent.

3.The probability distribution becomes very peaked abd sampling becomes nearly deterministic.
The model almost always will choose the highest-probability token, and the output will be very consistent but less diverse.   

"""
# ==============


# ==============
# Part 2 answers

PART2_CUSTOM_DATA_URL = None


def part2_gan_hyperparams():
    hypers = dict(
        batch_size=0, 
        z_dim=0, 
        learn_rate=0.0,
        betas=(0.0, 0.0), 
        disdiscriminator_optimizer={}, 
        generator_optimizer={},
        data_label=0,
        label_noise=0.0

    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
    hypers["batch_size"] =  128
    hypers["z_dim"] = 100
    hypers["learn_rate"] =  0.001
    hypers["betas"] = (0.4, 0.8)
    hypers["data_label"] = 0
    hypers["label_noise"] = 0.1
    hypers["discriminator_optimizer"] = {'type' : 'Adam', 
                                        'betas' : hypers["betas"],
                                        'lr' : hypers["learn_rate"]}
    hypers["generator_optimizer"] = {'type' : 'Adam',
                                        'betas' : hypers["betas"],
                                        'lr' : hypers["learn_rate"]}
    # ========================
    return hypers


part2_q1 = r"""
**Your answer:**


"""

part2_q2 = r"""
**Your answer:**


"""

part2_q3 = r"""
**Your answer:**



"""

part2_q4 = r"""
**Your answer:**


"""

# ==============

# ==============
# Part 3 answers

PART3_CUSTOM_DATA_URL = None


def part3_transformer_encoder_hyperparams():
    hypers = dict(
        embed_dim=128,    
        num_heads=8,          
        num_layers=6,         
        hidden_dim=512,       
        window_size=16,       
        dropout=0.1,         
        lr=1e-4,
    )

    # TODO: Tweak the hyperparameters to train the transformer encoder.
    # ====== YOUR CODE: ======

    # ========================
    return hypers


part3_q1 = r"""
**Your answer:**

"""

part3_q2 = r"""
**Your answer:**


"""

part4_q1 = r"""
**Your answer:**


"""

part4_q2 = r"""
**Your answer:**


"""

part4_q3 = r"""
**Your answer:**


"""

part4_q4 = r"""
**Your answer:**


"""

part4_q5 = r"""
**Your answer:**


"""

# ==============
