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
    hypers["batch_size"] =  64
    hypers["z_dim"] = 128
    hypers["learn_rate"] =  0.0002
    hypers["betas"] = (0.5, 0.999)
    hypers["data_label"] = 0
    hypers["label_noise"] = 0.15
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
The decision to maintain or discard gradients during sampling depends on the phase of training and the purpose of the sampling.

When we're updating the generator's parameters, we need to maintain gradients. This is because the generator's weights are updated based on the gradients of the loss calculated using the generated samples.

When updating the discriminator, we don't need gradients for the generated samples.
this is because at this stage we generate fake samples from random noise, pass real and fake samples through the discriminator, calculate the loss and backpropagate the loss and update only the discriminator's parameters.
We can discard gradients for the generated samples because we're not updating the generator in this step.

In addition, During inference, we want to generate samples without updating the model's weights, so we discard the gradients.



"""

part2_q2 = r"""
**Your answer:**

We should not stop training solely based on the Generator loss being below a threshold.
Loss values alone don't tell the full story. the generator and discriminator are in a constant competition. A low generator loss could simply mean that the generator has found a way to fool the current discriminator, not necessarily that it's producing high-quality or diverse images.
The generator loss doesn't directly correlate with the quality or diversity of the generated images because it's just a measure of how well the generator is fooling the current discriminator.

in addition GANs can suffer from mode collapse, where the generator produces a limited variety of samples that fool the discriminator but don't represent the full diversity of the target distribution.
A low generator loss could indicate that the generator is winning too easily, which might mean the discriminator isn't providing useful feedback anymore.

Discriminator loss remaining constant while generator loss decreases could indicate several things.
The discriminator might have become too good at its task, always correctly classifying real and fake images. This leads to a constant loss.
Meanwhile, the generator is still improving, managing to produce increasingly convincing fakes, hence its decreasing loss.

Secondly, mode collapse could be accuring, The generator might be focusing on producing a limited set of images that consistently fool the discriminator.
The discriminator's performance doesn't improve because it's seeing the same types of fake images, while the generator gets better at producing this limited set.


"""

part2_q3 = r"""
**Your answer:**



"""





# ==============

# ==============
# Part 3 answers

PART3_CUSTOM_DATA_URL = None


def part3_transformer_encoder_hyperparams():
	hypers = dict(
		embed_dim = 256,    
        num_heads = 8,           
        num_layers = 8,          
        hidden_dim = 1024,        
        window_size = 32,        
        dropout = 0.1,        
        lr = 0.00005,
	)

	# TODO: Tweak the hyperparameters to train the transformer encoder.
	# ====== YOUR CODE: ======

	# ========================
	return hypers


part3_q1 = r"""
**Your answer:**
Each layer processes the dependencies within the window. When we stack the layers the windows overlap,
and the information propegates accross wider spans.  
"""

part3_q2 = r"""
**Your answer:**
Instead of taking continuous window, we will take dilated window(for example if we are in the i-th step,
instead of taking the (i - w/2) to the (i + w/2) tokens, we will take the same number of tokens but with gaps between them).
This way the complexity stays the same but we have more global context.

"""

part4_q1 = r"""
**Your answer:**
BERT demonstrated significantly better results compared to our previously trained Transformer, whether fine-tuning just the last two linear layers or the entire model. This improvement is largely due to BERT being a much larger model that has been pre-trained on a vast dataset, giving it a more comprehensive understanding of the English language than what could be achieved with our smaller model, which was trained on a limited dataset and only uses windowed attention.

However, this advantage doesn't necessarily extend to all downstream tasks. For instance, in tasks requiring highly specialized knowledge, such as legal document classification or medical terminology identification, BERT's general language understanding might not be as effective as a model trained specifically on domain-specific data. A smaller model trained from scratch on a legal or medical dataset could potentially outperform BERT because it is tailored to the unique language and patterns of those fields.

Another example is in real-time, low-latency applications, such as voice-activated assistants or real-time translation systems. BERT's large size and complexity might make it less suitable for these tasks due to its higher computational demands. In these scenarios, a smaller, more efficient model trained from scratch for speed and responsiveness might be preferable, even if it doesn't match BERT's accuracy on more general tasks.
"""

part4_q2 = r"""
**Your answer:**

When the final two linear layers are frozen and only two internal layers are fine-tuned, the model's performance will decrease.

In BERT, the final linear layers are primarily responsible for classification and are closely tied to the specific task. Internal layers, which include multi-headed attention mechanisms, focus on general language understanding and are useful across a variety of tasks.

Fine-tuning internal layers is more challenging than adjusting the final classification layers for a few reasons. First, internal layers are positioned further from the output layer, so changes in them can lead to significant  unknown fluctuations in the model’s predictions. Additionally, while these layers do build upon the context provided by earlier layers, they might not immediately align with the specific task requirements, making it harder for them to accurately interpret nuanced aspects of a task, such as determining the sentiment of a review.

"""

part4_q3 = r"""
**Your answer:**

BERT was not originally designed for machine translation. BERT operates primarily as a masked language model without generative abilities. To enable BERT to perform translation tasks, many changes are necessary. 

a decoder whould need to be incorporated to generate text in the target language, thus converting BERT into a sequence-to-sequence model. Additionally, the pre-training process would need to be different to incorporate tasks specifically geared toward translation, such as denoising auto-encoding across various languages or predicting parallel sentences.

These changes are essential because effective translation requires both an understanding of the source language—where BERT excels—and the ability to generate accurate text in the target language, a capability absent in BERT's original design.
"""

part4_q4 = r"""
**Your answer:**

Choosing an RNN over a Transformer is often advantageous for tasks that involve sequences with strong temporal dependencies or when the input sequences are highly variable and potentially very long.

RNNs are inherently designed to process sequential data, maintaining the order of inputs, which makes them suitable for scenarios where the output depends significantly on prior inputs. Additionally, RNNs can handle sequences of various lengths with a consistent memory usage, which is beneficial when computational resources are limited or when dealing with extremely long sequences. 
In contrast, Transformers rely on self-attention mechanisms that scale quadratically with sequence length, which can be less efficient for very lengthy sequences.
"""

part4_q5 = r"""
**Your answer:**
Next Sentence Prediction (NSP) is a pre-training task in BERT where the model determines if one sentence follows another in the original text. During training, BERT processes both sentences together and uses the CLS token to make this prediction, with binary cross-entropy as the loss function. While NSP can help the model better understand the relationship between sentences and is useful for tasks like question answering, it's not absolutely essential. 
BERT can still perform effectively without it, as other tasks contribute more significantly to the model's ability to grasp context.

"""

# ==============
