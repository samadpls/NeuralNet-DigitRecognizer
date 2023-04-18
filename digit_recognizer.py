import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

@st.cache_data()
def get_data():
    with np.load('./data/mnist.npz') as f:
        imgs, labels = f['x_train'], f['y_train']
    imgs = imgs.astype('float32') / 255 
    imgs = np.reshape(imgs, (imgs.shape[0], imgs.shape[1] * imgs.shape[2]))
    labels = np.eye(10)[labels]
    return imgs, labels

def main():
    st.title("Digit Recognizer")

    images, labels = get_data()

   
#    w = weights, b = bias, i = input, h = hidden, o = output, l = label

    def fwd2hidden(img):
        h_pre = np.dot((b_i_h + w_i_h), img)
        return 1 / (1 + np.exp(-h_pre))

    def fwd2output(h):  
        o_pre = np.dot((b_h_o + w_h_o),  h)
        return 1 / (1 + np.exp(-o_pre))

    def errorCost(o):
        return 1 / len(o) * np.sum((o - l) ** 2, axis=0)

    w_i_h = np.random.uniform(-0.5, 0.5, (20, 784))
    w_h_o = np.random.uniform(-0.5, 0.5, (10, 20))
    b_i_h = np.zeros((20, 1))
    b_h_o = np.zeros((10, 1))

    learn_rate = 0.01
    nr_correct = 0
    epochs = 3
    # p = 0
    # progress_bar = st.progress(0)
    for epoch in range(epochs):
        for i, (img, l) in enumerate(zip(images, labels)):
            img.shape += (1,)
            l.shape += (1,)
            h = fwd2hidden(img)
            o = fwd2output(h)
            e = errorCost(o)

            nr_correct += int(np.argmax(o) == np.argmax(l))

            delta_o = o - l
            w_h_o += np.dot((-learn_rate * delta_o), np.transpose(h))
            b_h_o += -learn_rate * delta_o
            delta_h = np.dot(np.transpose(w_h_o), delta_o) * (h * (1 - h))
            w_i_h += np.dot(-learn_rate * delta_h, np.transpose(img))
            b_i_h += -learn_rate * delta_h
           
            # progress_text = "`Operation in progress. Please wait.`"

            # progress_bar.progress( ,text=progress_text)
           

        st.success(f"Acc: {round((nr_correct / images.shape[0]) * 100, 2)}%")
        nr_correct = 0

    try:
       
        index = st.text_input(f"Enter index for image (0 - 59999):")
        img = images[int(index)]
        fig, ax = plt.subplots()
        ax.imshow(img.reshape(28, 28), cmap="Greys")

        img.shape += (1,)
        # Forward propagation input -> hidden
        h_pre = np.dot((b_i_h + w_i_h), img.reshape(784, 1))
        h = 1 / (1 + np.exp(-h_pre))
        # Forward propagation hidden -> output
        o_pre = np.dot((b_h_o + w_h_o) , h)
        o = 1 / (1 + np.exp(-o_pre))

        ax.set_title(f"Number at {index} index is {o.argmax()} :)")
        st.pyplot(fig)
    except Exception as e:
        st.error("Must be number b/w 0-5999")

if __name__ == "__main__":
    main()
