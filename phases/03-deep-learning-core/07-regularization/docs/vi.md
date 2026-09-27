# Regularization

> Mô hình của bạn đạt 99% trên training data và 60% trên test data. Nó đã ghi nhớ thay vì học. Regularization là khoản thuế bạn áp đặt lên sự phức tạp để ép buộc generalization.

**Type:** Build
**Languages:** Python
**Prerequisites:** Lesson 03.06 (Optimizers)
**Time:** ~75 minutes

## Learning Objectives

- Triển khai dropout với inverted scaling, L2 weight decay, batch normalization, layer normalization, và RMSNorm từ đầu (from scratch)
- Đo lường train-test accuracy gap và chẩn đoán overfitting bằng các thí nghiệm regularization
- Giải thích tại sao transformers sử dụng LayerNorm thay vì BatchNorm và tại sao các LLM hiện đại ưu tiên RMSNorm
- Áp dụng sự kết hợp đúng đắn các kỹ thuật regularization dựa trên mức độ nghiêm trọng của overfitting

## The Problem

Một neural network với đủ parameters có thể ghi nhớ bất kỳ dataset nào. Đây không phải là giả thuyết -- Zhang et al. (2017) đã chứng minh điều đó bằng cách huấn luyện các standard networks trên ImageNet với random labels. Các networks đã đạt được training loss gần bằng 0 trên các random label assignments hoàn toàn ngẫu nhiên. Chúng đã ghi nhớ một triệu random input-output pairs mà không có pattern nào để học. Training loss hoàn hảo. Test accuracy bằng không.

Đây là vấn đề overfitting, và nó trở nên tồi tệ hơn khi các mô hình lớn hơn. GPT-3 có 175