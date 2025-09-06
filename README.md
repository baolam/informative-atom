#### Dự án hạt thông tin

##### Một số định hướng ngâm cứu

- Meta-representation dùng hướng tiếp cận hạt thông tin (tiềm năng, đào tạo trên bộ dữ liệu lớn nhỏ đều được)
- Nguyên lí phân rã vấn đề (một vấn đề bất kì, làm sao có thể phân rã nó ra thành các vấn đề con và phối hợp lời giải giữa chúng cho bài toán lớn (nguyên lí divide-and-conquer))
- Phân rã và tái kết nối (các thành phần có thể tự phối hợp với nhau cho vấn đề và rùi rã ra khi không sử dụng)
- Tính tương đồng của kiến trúc với Transformers (đặc điểm triển khai của Attention) và mạng Hopefield
- Tích hợp lý luận Logic cho biệt hoá đơn vị và cải thiện năng lực mô hình.

##### Lĩnh vực áp dụng của hạt thông tin

- Theo luật số lớn, nếu như không gian hạt là đủ lớn, không gian vấn đề là đủ lớn thì ta có thể hình thành biểu diễn có ý nghĩa đối với đầu vào chưa gặp hoặc rất lạ thông qua diễn giải đầu vào dưới các hạt đã biết.
- Tham vọng của ngâm cứu là có thể triển khai áp dụng vào tất cả các lĩnh vực và hình thành nên kho lời giải tổng quan, cần thì huy động tham gia hạt cần thiết.

##### Mục tiêu dự án

Thử khảo sát ý tưởng về hạt thông tin qua một số:

- Ý tưởng về tính chồng chấp trạng thái của hạt. Mục tiêu của hạt là để triển khai cho giải quyết một vấn đề bất kỳ. Xem vấn đề dưới dạng một hạt, tính chồng lấp trạng thái cho phép triển khai cài đặt cách giải quyết bằng hình thức dùng mô hình và cả phương pháp lập trình bằng thuật toán không phải AI.
- Đề xuất kiến trúc linh hoạt có thể lắp ghép, thay đổi theo từng vấn đề. Nghĩa là tuỳ vào vấn đề và họ vấn đề cần giải quyết, ta tự huy động ra kiến trúc phù hợp cho họ đó và cho phép quá trình tối ưu diễn ra trên kiến trúc đó. Khi cần giải quyết vấn đề mới, ta chỉ cần lắp những thành phần mới và ít thành phần cũ cho vấn đề mới. (Ý tưởng về lắp ghép Lego)

##### Một số câu hỏi định hướng ngâm cứu

- Mô hình AI là một hộp đen, mình có thể quy hoạch nội tại kiến trúc thành những vùng đen nhỏ và nhất định để _tăng tính giải thích_ của nó lên không? Đây là câu hỏi về tính giải thích ở mặt kiến trúc mô hình và mối quan hệ giữa tính mơ hồ và tính rõ ràng. Nếu giải quyết được câu hỏi này thì ta đạt được tính giải thích mô hình là đặc điểm nội tại của kiến trúc.
- Một quy trình suy luận trong AI (thuật toán lan truyền) yêu cầu sự hoạt động gần như của toàn bộ kiến trúc.Có cách nào mà định hướng theo vấn đề chỉ phần nào _thiết yếu mới cần hoạt động_ hay không? Định hướng câu hỏi này xuất phát trong Dynamic Network.
- Việc triển khai các góc nhìn khác nhau để đưa ra kiến trúc có dẫn đến mâu thuẫn tồn tại trong chính kiến trúc hay không? Câu hỏi này ở bối cảnh rộng hơn, mình có thể lý giải hành vi và áp dụng các kiến thức từ các lĩnh vực sinh học, xã hội, toán học, triết học vào hiểu và triển khai kiến trúc hay không? Câu hỏi này nếu giải quyết được thì nó có ý nghĩa trong dung hợp kiến thức các lĩnh vực.
- Với người thuần là lập trình thì họ có thể đóng góp gì cho kiến trúc?

##### Nguyên lí thiết kế

- Mô hình giải quyết vấn đề: Hiểu vấn đề --> Áp dụng giải quyết
  Quá trình hiểu vấn đề triển khai chi tiết gồm có từ các biểu hiện vấn đề xác định ra các tính chất và các biểu hiện tính chất của nó. Áp dụng giải quyết là tổng hợp các biểu hiện tính chất và diễn giải kết quả cho có ý nghĩa với vấn đề.

- Thiết kế gồm hai cái riêng biệt:
  Thiết kế kiến trúc là thiết kế ra cách mà các hạt tương tác với nhau.
  Thiết kế hành vi hạt là định nghĩa ra vai trò và gán ý nghĩa cho hạt đó.

- Nếu kiến trúc quá linh hoạt thì không thể áp dụng vào giải quyết vấn đề do tính hỗn loạn của nó. Kiến trúc cho một họ vấn đề (các vấn đề tương tự nhau) là tĩnh, không thay đổi nhưng có thể phân rã cho các bài toán mới.

- Hành vi của một hạt được định nghĩa thông qua bộ nhớ được đính kèm trong hạt. Ở góc nhìn sinh học, có thể xem bộ nhớ này như chất liệu di truyền. Các cách thiết kế hành vi phải dựa trên bộ nhớ.

- Tương tác hạt là cách định nghĩa luồng thông tin lan truyền qua các hạt như thế nào và vai trò vị trí của hạt trong vấn đề đó.

- Hiện tại các đơn vị tương tác với nhau khá là rời rạc, khi triển khai cho bài toán cụ thể, ta phải nhóm các đơn vị tương đồng với nhau thành một khối liền mạch.


##### Cấu trúc thư mục
docs --> chứa các ảnh minh hoạ kiến trúc cài đặt
experiment --> chứa kết quả thực nghiệm
fgi --> chứa mã nguồn chính của nghiên cứu
notebooks --> chứa chương trình chạy mã thực nghiệm