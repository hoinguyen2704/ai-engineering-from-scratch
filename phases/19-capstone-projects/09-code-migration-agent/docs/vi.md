# Capstone 09 — Code Migration Agent (Nâng cấp Runtime / Ngôn ngữ ở cấp độ Repo)

> MigrationBench của Amazon (Java 8 lên 17) và trình di chuyển App Engine Py2-sang-Py3 của Google đã thiết lập tiêu chuẩn cho năm 2026. OpenRewrite của Moderne thực hiện các thao tác viết lại AST (Abstract Syntax Tree) có tính quyết định ở quy mô lớn. Grit nhắm đến cùng vấn đề này với DSL theo phong cách codemod. Mô hình sản xuất kết hợp cả hai: một nền tảng có tính quyết định (deterministic substrate) cho các thao tác viết lại an toàn, cộng với một lớp agent cho các trường hợp mơ hồ, một sandbox cho các bản build theo từng nhánh, và một bộ kiểm thử (test harness) đảm bảo trạng thái "xanh" trước khi mở PR. Capstone này yêu cầu bạn di chuyển 50 repo thực tế và công bố tỷ lệ thành công cùng với phân loại lỗi (failure taxonomy).

**Type:** Capstone
**Languages:** Python (agent), Java / Python (targets), TypeScript (dashboard)
**Prerequisites:** Phase 5 (NLP), Phase 7 (transformers), Phase 11 (LLM engineering), Phase 13 (tools), Phase 14 (agents), Phase 15 (autonomous), Phase 17 (infrastructure)
**Phases exercised:** P5 · P7 · P11 · P13 · P14 · P15 · P17
**Time:** 30 giờ

## Vấn đề

Di chuyển mã nguồn quy mô lớn là một trong những ứng dụng thực tế rõ ràng nhất của các coding agent năm 2026. Ground truth rất hiển nhiên (liệu bộ test có vượt qua sau khi di chuyển không?), phần thưởng rất thực tế (di chuyển một đội tàu Java-8 là một dự án quy mô lớn), và các benchmark đều công khai (tập con 50-repo của MigrationBench). OpenRewrite của Moderne xử lý phía có tính quyết định. Lớp agent xử lý mọi thứ mà các công thức (recipes) của OpenRewrite không làm được: các thao tác viết lại mơ hồ, sự trôi dạt của hệ thống build (build-system drift), cú pháp hiếm gặp, và lỗi hỏng phụ thuộc bắc cầu (transitive dependency).

Bạn sẽ xây dựng một agent nhận vào một repo Java 8 (hoặc Python 2) và tạo ra một nhánh đã di chuyển với CI xanh. Bạn sẽ đo lường tỷ lệ thành công, khả năng bảo toàn độ bao phủ kiểm thử (test-coverage), chi phí trên mỗi repo, và xây dựng một phân loại lỗi. Việc so sánh song song với một baseline chỉ dùng phương pháp có tính quyết định sẽ cho bạn thấy giá trị thực sự của agent nằm ở đâu.

## Khái niệm

Pipeline có hai lớp. **Nền tảng có tính quyết định** (OpenRewrite cho Java, libcst cho Python) chạy phần lớn các thao tác viết lại cơ học một cách an toàn: import, chữ ký phương thức, chỉnh sửa null-safety, try-with-resources, thay thế API đã lỗi thời. Nó nhanh và tạo ra các diff có thể kiểm toán được. **Lớp agent** (OpenAI Agents SDK hoặc LangGraph trên Claude Opus 4.7 và GPT-5.4-Codex) xử lý các trường hợp mà công thức không thể: nâng cấp file build (Maven/Gradle/pyproject), xung đột phụ thuộc bắc cầu, test flakes, các annotation tùy chỉnh.

Mỗi repo nhận được một sandbox Daytona với runtime mục tiêu đã được cài đặt sẵn. Agent lặp lại quy trình: chạy build, phân loại lỗi, áp dụng bản sửa lỗi, chạy lại. Giới hạn cứng: 30 phút mỗi repo, $8 mỗi repo, 20 lượt agent. Nếu tất cả các test đều vượt qua và độ bao phủ không bị giảm, nhánh sẽ mở một PR. Nếu không, repo sẽ được xếp vào một lớp lỗi kèm theo bằng chứng.

Phân loại lỗi là sản phẩm bàn giao. Qua 50 repo, cái gì đã hỏng? Phụ thuộc bắc cầu? Annotation tùy chỉnh? Phiên bản công cụ build? Test flakes không liên quan đến di chuyển? Mỗi lớp sẽ có số lượng và một diff mẫu. Các tác giả viết công thức trong tương lai có thể nhắm vào ba lỗi hàng đầu.

## Kiến trúc

```
target repo
      |
      v
OpenRewrite / libcst deterministic recipes
   (safe, fast, auditable, ~70-80% of fixes)
      |
      v
Daytona sandbox per branch
      |
      v
agent loop (Claude Opus 4.7 / GPT-5.4-Codex):
   - run build -> capture failures
   - classify failures (build, test, lint)
   - apply fix (patch or retry recipe)
   - rerun
   - budget: 30 min, $8, 20 turns
      |
      v
test + coverage delta gate
      |
      v (passed)
open PR
      |
      v (failed)
file under failure class + attach repro
```

## Stack

- Nền tảng có tính quyết định: OpenRewrite (Java) hoặc libcst (Python)
- Agent: OpenAI Agents SDK hoặc LangGraph trên Claude Opus 4.7 + GPT-5.4-Codex
- Sandbox: Daytona devcontainers cho mỗi nhánh, cài đặt sẵn runtime mục tiêu (Java 17 / Python 3.12)
- Hệ thống build: Maven, Gradle, uv (Python)
- Benchmarks: Tập con 50-repo của Amazon MigrationBench (Java 8 lên 17), các repo Py2-sang-Py3 của Google App Engine
- Test harness: trình chạy song song, đo độ bao phủ qua Jacoco (Java) hoặc coverage.py (Python)
- Khả năng quan sát: Langfuse + trace bundle cho mỗi repo với mọi đoạn diff
- Dashboard: dashboard phân loại lỗi với số lượng theo lớp và các diff mẫu

```figure
ce-migration-funnel
```

## Xây dựng

1. **Chạy công thức.** Chạy các công thức OpenRewrite (Java) hoặc libcst (Python) trước. Bắt lấy 70-80% các ca di chuyển mang tính cơ học. Commit dưới dạng "recipe" commit.

2. **Thử nghiệm build.** Sandbox Daytona: cài đặt runtime mục tiêu, chạy build. Nếu xanh, chuyển sang test. Nếu đỏ, chuyển giao cho agent.

3. **Vòng lặp agent.** LangGraph với các công cụ: `run_build`, `read_file`, `edit_file`, `run_test`, `git_diff`. Agent phân loại lỗi (phụ thuộc, cú pháp, test, công cụ build) và áp dụng bản sửa lỗi mục tiêu. Chạy lại.

4. **Giới hạn ngân sách.** 30 phút thời gian thực mỗi repo, $8 chi phí, 20 lượt agent. Bất kỳ sự vi phạm nào cũng sẽ dừng lại và xếp vào "budget_exhausted" với diff hiện tại.

5. **Cổng kiểm thử + độ bao phủ.** Sau khi build xanh, chạy bộ test. So sánh độ bao phủ với repo gốc. Nếu độ bao phủ giảm hơn 2%, xếp vào "coverage_regression".

6. **Mở PR.** Khi thành công, push nhánh, mở PR với diff và tóm tắt các công thức đã áp dụng và các commit mà agent đã tạo.

7. **Phân loại lỗi.** Với mỗi repo thất bại, gắn nhãn với một lớp: `dep_upgrade_required`, `build_tool_drift`, `custom_annotation`, `test_flake`, `syntax_edge_case`, `budget_exhausted`. Xây dựng dashboard.

8. **Chạy trên 50 repo.** Thực thi trên tập con MigrationBench. Báo cáo tỷ lệ thành công theo lớp, chi phí mỗi repo, bảo toàn độ bao phủ, và so sánh với baseline chỉ dùng phương pháp có tính quyết định.

## Sử dụng

```
$ migrate legacy-java-service --target java17
[recipe]   27 rewrites applied (JUnit 4->5, HashMap initializer, try-with-resources)
[build]    FAIL: cannot find symbol sun.misc.BASE64Encoder
[agent]    turn 1 classify: removed_jdk_api
[agent]    turn 2 apply: sun.misc.BASE64Encoder -> java.util.Base64
[build]    OK
[tests]    412/412 passing; coverage 84.1% -> 84.3%
[pr]       opened #1841  cost=$3.20  turns=4
```

## Giao hàng

`outputs/skill-migration-agent.md` là sản phẩm bàn giao. Với một repo, nó thực thi các công thức có tính quyết định sau đó là vòng lặp agent để tạo ra một nhánh đã di chuyển thành công, hoặc xếp repo vào một lớp phân loại lỗi.

| Trọng số | Tiêu chí | Cách đo lường |
|:-:|---|---|
| 25 | Tỷ lệ thành công MigrationBench | pass@1 trên tập con 50-repo |
| 20 | Bảo toàn độ bao phủ kiểm thử | Độ lệch độ bao phủ trung bình so với gốc |
| 20 | Chi phí mỗi repo đã di chuyển | $/repo trên các lần chạy thành công |
| 20 | Tích hợp agent / công cụ quyết định | Tỷ lệ các bản sửa lỗi do OpenRewrite xử lý so với do agent tạo |
| 15 | Viết phân tích lỗi | Độ đầy đủ của phân loại kèm theo các mẫu |
| **100** | | |

## Bài tập

1. Chạy pipeline di chuyển chỉ với OpenRewrite (không có agent). So sánh tỷ lệ thành công với pipeline đầy đủ. Xác định các trường hợp mà chỉ có agent mới tạo ra sự khác biệt.

2. Triển khai kiểm tra "lint-clean": sau khi di chuyển, chạy một linter kiểu dáng (spotless cho Java, ruff cho Python). Làm thất bại PR nếu xuất hiện lỗi lint mới. Đo tỷ lệ bảo toàn độ bao phủ nhưng bị thoái hóa kiểu dáng.

3. Thêm trình tối ưu hóa "minimal-diff": sau khi nhánh của agent vượt qua các bài test, cắt tỉa các thay đổi không cần thiết bằng một lượt chạy thứ hai. Báo cáo mức độ giảm kích thước diff.

4. Mở rộng sang lần di chuyển thứ ba: Node 18 lên Node 22. Tái sử dụng việc bao bọc sandbox; thay thế lớp công thức bằng một codemod tùy chỉnh.

5. Đo lường thời gian đến bản build xanh đầu tiên (TTFGB) như một chỉ số UX. Mục tiêu: p50 dưới 10 phút.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|-----------------|------------------------|
| Nền tảng có tính quyết định | "Recipe engine" | OpenRewrite / libcst: viết lại AST khai báo với các đảm bảo an toàn |
| Codemod | "Code-modifying program" | Một quy tắc viết lại thay đổi mã nguồn một cách cơ học |
| Build drift | "Tool version skew" | Sự thay đổi hành vi tinh vi của Maven / Gradle / uv giữa các phiên bản chính |
| Lớp lỗi | "Taxonomy bucket" | Lý do được gắn nhãn khiến repo không di chuyển được: phụ thuộc, cú pháp, test, công cụ build, ngân sách |
| Độ lệch độ bao phủ | "Coverage preservation" | Thay đổi % độ bao phủ kiểm thử từ nhánh gốc sang nhánh đã di chuyển |
| Lượt agent | "Tool-call round" | Một chu kỳ lập kế hoạch -> hành động -> quan sát trong vòng lặp agent |
| Cạn kiệt ngân sách | "Hit the ceiling" | Repo đã tiêu tốn giới hạn 30 phút / $8 / 20 lượt mà không thành công |

## Đọc thêm

- [Amazon MigrationBench](https://aws.amazon.com/blogs/devops/amazon-introduces-two-benchmark-datasets-for-evaluating-ai-agents-ability-on-code-migration/) — benchmark chuẩn năm 2026
- [Nền tảng Moderne.io OpenRewrite](https://www.moderne.io) — tài liệu tham khảo về nền tảng có tính quyết định
- [Tài liệu OpenRewrite](https://docs.openrewrite.org) — viết công thức
- [Grit.io](https://www.grit.io) — DSL codemod thay thế
- [Sách hướng dẫn di chuyển trong sandbox của OpenAI](https://developers.openai.com/cookbook/examples/agents_sdk/sandboxed-code-migration/sandboxed_code_migration_agent) — tài liệu tham khảo Agents SDK
- [Trình di chuyển Py2 sang Py3 của Google App Engine](https://cloud.google.com/appengine) — benchmark di chuyển thay thế
- [libcst](https://github.com/Instagram/LibCST) — nền tảng có tính quyết định cho Python
- [Sandbox Daytona](https://daytona.io) — sandbox tham khảo cho mỗi nhánh