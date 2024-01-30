
from deep_visualization.grad_cam import GradCAM, overlay_gradCAM
from deep_visualization.guided_backprob import GuidedBackprop, deprocess_image
import matplotlib.pyplot as plt
import numpy as np
from deep_visualization.guided_bp import *





def preprocess(img_path):
    # im = img_to_array(load_img(os.path.join(img_path, filename), target_size=TARGET_SIZE))
    # x = np.expand_dims(im, axis=0)
    # x = preprocess_input(x)
    image = cv2.imread(img_path)
    image = np.array(image) / 255
    b, g, r = cv2.split(image)
    image = cv2.merge([r, g, b])
    x = np.expand_dims(image, axis=0)

    return x

def predict(model, processed_im):
    preds = model.predict(processed_im)
    idx = preds.argmax()
    res = [idx, preds.max()]
    print("Predicted:", idx)
    # predictions = model(processed_im, training=False).numpy()
    # print("Predicted:", predictions)
    # idx = (predictions > 0.5).astype(int)
    # res = [idx[0, 0], predictions[0, 0]]
    return res




def showCAMs(img, x, GradCAM, GuidedBP, chosen_class, upsample_size):

    # Grad-CAM
    cam3 = GradCAM.compute_heatmap(image=x, classIdx=chosen_class, upsample_size=upsample_size)
    gradcam = overlay_gradCAM(img, cam3)
    gradcam = cv2.cvtColor(gradcam, cv2.COLOR_BGR2RGB)
    # plt.imshow(gradcam)
    # plt.show()

    # Guided backprop
    # gb = GuidedBP.guided_backprop(x, upsample_size)
    # gb_im = deprocess_image(gb)
    # gb_im = cv2.cvtColor(gb_im, cv2.COLOR_BGR2RGB)
    gb = start()
    gb_im = deprocess_image(gb)
    gb_im = cv2.cvtColor(gb_im, cv2.COLOR_BGR2RGB)

    # plt.imshow(gb_im)
    # plt.show()

    # Guided GradCAM
    guided_gradcam = deprocess_image(gb * cam3)
    guided_gradcam = cv2.cvtColor(guided_gradcam, cv2.COLOR_BGR2RGB)
    # plt.imshow(guided_gradcam)
    # plt.show()

    # # Display
    # 设置画布大小
    plt.figure(figsize=(15, 5))

    # 调整子图之间的间距
    plt.subplots_adjust(wspace=0.3)

    # 显示 Grad-CAM
    plt.subplot(1, 3, 1)
    plt.imshow(gradcam)
    plt.title("Grad-CAM")

    # 显示 Guided backprop
    plt.subplot(1, 3, 2)
    plt.imshow(gb_im)
    plt.title("Guided Backpropagation")

    # 显示 Guided GradCAM
    plt.subplot(1, 3, 3)
    plt.imshow(guided_gradcam)
    plt.title("Guided Grad-CAM")

    # 显示图形
    plt.show()


    # cv2.imwrite("Heatmap.jpg", cam * 255)




def deep_visualization(model, img_path ):
    # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    gradCAM = GradCAM(model=model, layerName="conv2d_3")   # "conv2d_9"
    guidedBP = GuidedBackprop(model=model)

    #     img = img_to_array(load_img(os.path.join(SAMPLE_DIR,imgs.value), target_size=(224,224)))

    img = cv2.imread(img_path)
    upsample_size = (img.shape[1], img.shape[0])
    x = preprocess(img_path)

    pred, prob = predict(model, x)
    classIdx = pred
    print(classIdx)

    showCAMs(img, x, gradCAM, guidedBP, classIdx, upsample_size)


