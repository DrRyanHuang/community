# 一起从代码层面熟悉 CINN 编译器 —— CINN 源码阅读指南

| 版本  | 作者  | 指导/校验  | 时间       | 主要更新           |
| ---- | ----- | --------- |---------- | ------------------ |
| 0.1 | [Ryan](https://github.com/drryanhuang)  | [Aurelius84](https://github.com/Aurelius84)      |2023.10.28  | 初版|

阅读本指南的收获

- 阅读 Paddle 工业级源码, 提升 C++ 编码能力
- 从代码层面了解 CINN 编译器的设计
- 熟悉 Paddle 代码风格
- ~~学习胡乱使用倒叙和插叙的混乱啰嗦的博客风格写作手法~~


由于 PIR 和 CINN 依旧在高频更新, 如果笔者写的有问题, 各位开发者大佬提个PR帮我修改一下, 万分感谢! 本指南也可以看做是对 [first_step.md](first_step.md) 的补充.

### 1. 开胃菜 pir_compiler_test 单测
 

一个小技巧, 看源码不知道从哪里开始看, 那就从单测开始吧！(杰哥[@Aurelius84](https://github.com/Aurelius84)告诉我的)
先来看一点儿简单的, `ProgramInfo` 是个 `tuple`, 不能通过下标获取元素, 而是通过 `std::get<i>` 来获取第 `i` 个元素

```c++
// test/cpp/pir/cinn/pir_compiler_test.cc
using ProgramInfo =
    std::tuple<std::shared_ptr<::pir::Program>, std::vector<GroupPtr>>;
  
  // ......

  std::shared_ptr<::pir::Program> program = std::get<0>(prog_info);
  std::vector<GroupPtr> groups = std::get<1>(prog_info);
```

两个返回的元素中, `Program` 在 [first_step.md](first_step.md) 部分已经提到过, 是模型结构的抽象，分为计算图 `graphs` 和权重 `weights`. `GroupPtr` 先入栈, 稍后再说.


`pir_compiler_test.cc` 文件共有4个 TEST:

```c++
// test/cpp/pir/cinn/pir_compiler_test.cc

TEST(PIRCompier, CompileSoftmax)
TEST(PIRCompier, CompilerAndRun)
TEST(PIRCompier, CompileGroupOps)
TEST(RuntimeDialect, CompilerAndRun)
```

`BuildProgram` 和 `BuildSoftmax` 两个函数用来生成 TEST 需要的 `ProgramInfo`. 两个函数前3行是一样的. 

```c++
  // test/cpp/pir/cinn/pir_compiler_test.cc

  // 创建 IrContext 单例
  ::pir::IrContext* ctx = ::pir::IrContext::Instance();
  // 从 IrContext 中获取 OperatorDialect 实例, 如果没有找到, 则创建并注册到 IrContext
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
  // 通过 IrContext 创建 Program
  auto program = std::make_shared<::pir::Program>(ctx);
```

`IrContext` 在 [first_step.md](first_step.md) 结尾部分提到过, `IrContext` 是一个全局无参数类，用于存储和管理 Type 和 Attribute 等相关数据结构，本小节重新来看一下.

模板函数 `GetOrRegisterDialect` 获取 `IrContext` 中的对应类的 `dialect`，如果没有找到，则创建并注册到 `IrContext`, 其中调用了其重载版本.

其模板函数通过传入不同的模板参数来实现多态, 其重载版本用来实现对应的功能.


```c++
  // paddle/pir/core/ir_context.h

  template <typename DialectT>
  DialectT *GetOrRegisterDialect() {
    return static_cast<DialectT *>(
        GetOrRegisterDialect(DialectT::name(), [this]() {
          DialectT *dialect = new DialectT(this);
          return dialect;
        }));
  }
```

`GetOrRegisterDialect` 的重载版本:

```c++ 
// paddle/pir/core/ir_context.cc

Dialect *IrContext::GetOrRegisterDialect(
    const std::string &dialect_name, std::function<Dialect *()> constructor) {
  VLOG(4) << "Try to get or register a Dialect of: [name=" << dialect_name
          << "].";
  
  // 查找该 Dialect 是否已经被注册
  if (!impl().IsDialectRegistered(dialect_name)) {
    VLOG(4) << "Create and register a new Dialect of: [name=" << dialect_name
            << "].";
    
    // 如果该 Dialect 未被注册, 则创建并注册
    impl().RegisterDialect(dialect_name, constructor());
  }

  // 若已注册则直接返回
  return impl().GetDialect(dialect_name);
}
```

看到 `impl()` 我们猜到 `IrContext` 也是 `pImpl` 的设计模式

```c++
  // paddle/pir/core/ir_context.h

  IrContextImpl &impl() { return *impl_; }
```

接下来简单看一下 `IsDialectRegistered`, `RegisterDialect` 和 `GetDialect` 三个函数.

因为 `registed_dialect_` 是个 `unordered_map`, 所以以上三个函数基本都是对 `unordered_map` 相关函数的封装.

```c++
  // paddle/pir/core/ir_context.cc

  // The dialect registered in the context.
  std::unordered_map<std::string, Dialect *> registed_dialect_;
  pir::SpinLock registed_dialect_lock_;

  // ......

  bool IsDialectRegistered(const std::string &name) {
    return registed_dialect_.find(name) != registed_dialect_.end();
  }

  void RegisterDialect(std::string name, Dialect *dialect) {
    std::lock_guard<pir::SpinLock> guard(registed_dialect_lock_);
    VLOG(6) << "Register a dialect of: [name=" << name
            << ", dialect_ptr=" << dialect << "].";
    registed_dialect_.emplace(name, dialect);
  }

  Dialect *GetDialect(const std::string &name) {
    std::lock_guard<pir::SpinLock> guard(registed_dialect_lock_);
    auto iter = registed_dialect_.find(name);
    if (iter != registed_dialect_.end()) {
      VLOG(6) << "Found a cached dialect of: [name=" << name
              << ", dialect_ptr=" << iter->second << "].";
      return iter->second;
    }
    LOG(WARNING) << "No cache found dialect of: [name=" << name << "].";
    return nullptr;
  }
```

因为 `IrContext` 是个单例, 多线程的情况下, 会产生线程不安全的情况, 所以需要加锁:

```c++
  // paddle/pir/core/ir_context.cc

  std::lock_guard<pir::SpinLock> guard(registed_dialect_lock_);
```

好的, 现在我们跳回去, 执行完 `GetOrRegisterDialect` 之后, `paddle::dialect::OperatorDialect` 被注册到 `IrContext` 中. 


这里来看 `paddle::dialect::OperatorDialect` 中到底注册了哪些信息, 其中依旧是注册了一系列的类型、属性和 Op 等, 属性和算子都是通过传入变长模板参数来实现的.

```c++
// paddle/fluid/pir/dialect/operator/ir/op_dialect.cc
void OperatorDialect::initialize() {

  // 注册类型
  RegisterTypes<paddle::dialect::DenseTensorType>();
  RegisterTypes<paddle::dialect::SelectedRowsType>();

  // 注册属性
  RegisterAttributes<paddle::dialect::IntArrayAttribute,
                     paddle::dialect::DataTypeAttribute,
                     paddle::dialect::PlaceAttribute,
                     paddle::dialect::DataLayoutAttribute>();

  // 注册一些算子 Op
  RegisterOps<
#define GET_OP_LIST
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"  // NOLINT
      >();
  RegisterOps<
#define GET_OP_LIST
#include "paddle/fluid/pir/dialect/operator/ir/control_flow_op.cc"  // NOLINT
      >();
  RegisterOps<paddle::dialect::AddNOp,
              paddle::dialect::AddN_Op,
              paddle::dialect::AddNWithKernelOp,
              paddle::dialect::FusedGemmEpilogueOp,
              paddle::dialect::FusedGemmEpilogueGradOp,
              paddle::dialect::SplitGradOp>();

  RegisterInterfaces<ParameterConvertInterface>();
}
```

与之前不同的是, 宏 `GET_OP_LIST` 使用了两次

```c++
  RegisterOps<
#define GET_OP_LIST
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"  // NOLINT
      >();
  RegisterOps<
#define GET_OP_LIST
#include "paddle/fluid/pir/dialect/operator/ir/control_flow_op.cc"  // NOLINT
      >();
```

`pd_op.h` 是通过 `op_gen.py` 生成的:

```c++
#ifdef GET_OP_LIST
#undef GET_OP_LIST
paddle::dialect::AbsOp, paddle::dialect::Abs_Op, paddle::dialect::AccuracyOp, paddle::dialect::AcosOp, paddle::dialect::Acos_Op, paddle::dialect::AcoshOp, paddle::dialect::Acosh_Op, paddle::dialect::Adagrad_Op, paddle::dialect::Adam_Op, paddle::dialect::Adamax_Op, paddle::dialect::Adamw_Op, paddle::dialect::AddmmOp, 
// ......

#else
// This file is generated by "paddle/fluid/pir/dialect/op_generator/op_gen.py"

#endif
```

通过 `#include` 文件, 来将对应的算子传给 `RegisterOps` 的模板参数. 之后就是传入 `ctx` 来创建 `Program`. 

```c++
auto program = std::make_shared<::pir::Program>(ctx);
::pir::Builder builder = ::pir::Builder(ctx, program->block());
```

`Builder` 的构造函数有多个, `Builder` 类是 `Attribute` 类的统一接口。 所有 `Attribute` 类的派生仅派生接口，而不派生成员。


```c++
// paddle/pir/core/builder.h

using InsertPoint = std::pair<Block *, Block::Iterator>;

  Builder(IrContext *context, Block *block, Block::Iterator insert_point)
      : context_(context), insert_point_(block, insert_point) {}

  Builder(IrContext *context, Block *block)
      : Builder(context, block, block->end()) {}
```

在 `BuildProgram` 函数中, 创建了一系列 Operation

```c++
  // test/cpp/pir/cinn/pir_compiler_test.cc

  const float value_one = 1.0;  // relu(tan(1.)) = 1.5;
  const float value_two = 2.0;  // relu(tan(2.)) = 0.
  auto full_op_x =
      builder.Build<paddle::dialect::FullOp>(std::vector<int64_t>{64, 128},
                                             value_one,
                                             phi::DataType::FLOAT32,
                                             phi::GPUPlace());

  auto full_op_y =
      builder.Build<paddle::dialect::FullOp>(std::vector<int64_t>{64, 128},
                                             value_two,
                                             phi::DataType::FLOAT32,
                                             phi::GPUPlace());

  auto tan_op_x = builder.Build<paddle::dialect::TanOp>(full_op_x->result(0));
  auto relu_op_x = builder.Build<paddle::dialect::ReluOp>(tan_op_x->result(0));
  auto tan_op_y = builder.Build<paddle::dialect::TanOp>(relu_op_x->result(0));
  auto relu_op_y = builder.Build<paddle::dialect::ReluOp>(tan_op_y->result(0));
```


`Builder::Build` 接受可变模版参数, 内部通过 `IrContext context_` 来创建 `OperationArgument`.

```c++
// paddle/pir/core/builder.h

template <typename OpTy, typename... Args>
OpTy Builder::Build(Args &&...args) {
  OperationArgument argument(context_->GetRegisteredOpInfo(OpTy::name()));
  OpTy::Build(*this, argument, std::forward<Args>(args)...);
  Operation *op = Build(std::move(argument));
  return OpTy(op);
}
```

之后调用当前 Op 的静态 `Build` 函数, 这里以 `FullOp::Build` 为例, 值得一提的是, `pd_op.cc` 和 `pd_op.h` 都是通过 `op_gen.py` 来生成的, 文件在 `build` 目录下. 

`FullOp::Build` 函数主要是修改 `OperationArgument &argument` 中的内容.


```c++
// build/paddle/fluid/pir/dialect/operator/ir/pd_op.cc

void FullOp::Build(pir::Builder &builder, pir::OperationArgument &argument, const std::vector<int64_t>& shape, float value, phi::DataType dtype, const Place& place) {
  VLOG(4) << "Start build FullOp";



  VLOG(4) << "Builder construction inputs";

  VLOG(4) << "Builder construction attributes";
  pir::Attribute attr_shape = paddle::dialect::IntArrayAttribute::get(pir::IrContext::Instance(), phi::IntArray(shape));
  argument.AddAttribute("shape", attr_shape);
  pir::Attribute attr_value = paddle::dialect::TransToIrAttribute(value, pir::IrContext::Instance());
  argument.AddAttribute("value", attr_value);
  pir::Attribute attr_dtype = paddle::dialect::DataTypeAttribute::get(pir::IrContext::Instance(), dtype);
  argument.AddAttribute("dtype", attr_dtype);
  pir::Attribute attr_place = paddle::dialect::PlaceAttribute::get(pir::IrContext::Instance(), place);
  argument.AddAttribute("place", attr_place);

  VLOG(4) << "Builder construction outputs";
  paddle::dialect::IrMetaTensor dense_out;
  phi::MetaTensor meta_out(&dense_out);

  phi::CreateInferMeta(shape, dtype, &meta_out);

  std::vector<pir::Type> argument_outputs;
  pir::Type out_dense_tensor_type = paddle::dialect::DenseTensorType::get(pir::IrContext::Instance(), paddle::dialect::TransToIrDataType(dense_out.dtype()), dense_out.dims(), dense_out.layout(), dense_out.lod(), dense_out.offset());
  argument_outputs.push_back(out_dense_tensor_type);
  argument.AddOutputs(argument_outputs.begin(), argument_outputs.end());
  ::pir::PassStopGradientsDefaultly(argument);

}
```


之后, 调用创建 `Operation` 的 `Build` 函数, 调用 `Insert` 函数, 来将刚刚创建的 `Operation` 实例指针, 插入到当前的 `Program` `Block` 中


```c++
// paddle/pir/core/builder.cc

/// Create an operation given the fields represented as an OperationState.
Operation *Builder::Build(OperationArgument &&argument) {
  return Insert(Operation::Create(std::move(argument)));
}

/// Creates an operation with the given fields.
Operation *Builder::Build(const std::vector<Value> &inputs,
                          const AttributeMap &attribute,
                          const std::vector<Type> &output_types,
                          OpInfo op_info) {
  return Build(OperationArgument(inputs, attribute, output_types, op_info));
}

Operation *Builder::Insert(Operation *op) {
  if (insert_point_.first) {
    insert_point_.first->insert(insert_point_.second, op);  // <-------- 这个 list.end() 不会变吗? 
  } else {
    LOG(WARNING) << "Builder's Block is nullptr, insert failed.";
  }
  return op;
}
```


最后调用基类的基类构造函数并返回 `OpTy(op)`.

```c++
// paddle/pir/core/op_base.h
explicit OpBase(Operation *operation = nullptr) : operation_(operation) {}
```


再来看一下静态函数 `ReluOp::Build`

```c++
// build/paddle/fluid/pir/dialect/operator/ir/pd_op.cc

void ReluOp::Build(pir::Builder &builder, pir::OperationArgument &argument, pir::Value x_) {
  VLOG(4) << "Start build ReluOp";



  VLOG(4) << "Builder construction inputs";
  std::vector<pir::Value> argument_inputs = {x_};
  argument.AddInputs(argument_inputs);

  VLOG(4) << "Builder construction attributes";

  VLOG(4) << "Builder construction outputs";
  paddle::dialect::DenseTensorType x = x_.type().dyn_cast<paddle::dialect::DenseTensorType>(); (void)x;  // <--------- 这里是?

  VLOG(4) << "Builder construction  dense_x";
  paddle::dialect::IrMetaTensor ir_meta_tensor_x(paddle::dialect::TransToPhiDataType(x.dtype()),
                                                      x.dims(),
                                                      x.data_layout(),
                                                      x.lod(),
                                                      x.offset());
  VLOG(4) << "Builder construction  meta_x";
  phi::MetaTensor meta_x(&ir_meta_tensor_x);
  paddle::dialect::IrMetaTensor dense_out;
  phi::MetaTensor meta_out(&dense_out);

  phi::UnchangedInferMeta(meta_x, &meta_out);

  std::vector<pir::Type> argument_outputs;
  pir::Type out_dense_tensor_type = paddle::dialect::DenseTensorType::get(pir::IrContext::Instance(), paddle::dialect::TransToIrDataType(dense_out.dtype()), dense_out.dims(), dense_out.layout(), dense_out.lod(), dense_out.offset());
  argument_outputs.push_back(out_dense_tensor_type);
  argument.AddOutputs(argument_outputs.begin(), argument_outputs.end());
  ::pir::PassStopGradientsDefaultly(argument);

}
```

参数传入了 `pir::Value x_`, 所以 `argument` 要添加输入 `{x_}`, 关于内部 `phi::MetaTensor` 部分, 我们之后再看

```c++
  std::vector<pir::Value> argument_inputs = {x_};
  argument.AddInputs(argument_inputs);
```

如果 Op 有多个输入呢? 我们来看下 `MatmulOp` 有两个输入.

```c++
// build/paddle/fluid/pir/dialect/operator/ir/pd_op.cc

void MatmulOp::Build(pir::Builder &builder, pir::OperationArgument &argument, pir::Value x_, pir::Value y_, bool transpose_x, bool transpose_y) {
  VLOG(4) << "Start build MatmulOp";



  VLOG(4) << "Builder construction inputs";
  std::vector<pir::Value> argument_inputs = {x_, y_};
  argument.AddInputs(argument_inputs);

  // ......

  phi::MatmulInferMeta(meta_x, meta_y, transpose_x, transpose_y, &meta_out);

  // ......
}
```

接下来终于到 `groups` 部分了, 其每一个元素是 `GroupPtr` 

```c++
  // paddle/cinn/hlir/framework/pir/op_lowering_impl.h
  using GroupPtr = std::shared_ptr<Group>;
  

  // test/cpp/pir/cinn/pir_compiler_test.cc
  std::vector<GroupPtr> groups;
  groups.emplace_back(
      std::make_shared<Group>(std::initializer_list<::pir::Operation*>(
          {full_op_x.operation()})));  // For coverage
  groups.emplace_back(std::make_shared<Group>(
      std::initializer_list<::pir::Operation*>({full_op_y.operation()})));
  groups.emplace_back(std::make_shared<Group>(
      std::vector<::pir::Operation*>({tan_op_x.operation(),
                                      relu_op_x.operation(),
                                      tan_op_y.operation(),
                                      relu_op_y.operation()})));
```

每个 `Group` 就是由一系列算子指针 `Operation*` 组成, 从其构造函数也可以看出来:

```c++
  // paddle/cinn/hlir/framework/pir/group.h

  explicit Group(const std::vector<::pir::Operation*>& group_ops)
      : ops(group_ops) {}

  explicit Group(std::initializer_list<::pir::Operation*> group_ops)
      : ops(group_ops) {}

  // ......
  std::vector<::pir::Operation*> ops;
```

所以 `BuildProgram` 函数返回了刚刚建立的 `program` 和 `groups`.



三个单测都有这三句代码来打印整个计算图, 我们也来顺便看一下, 其实就是依次调用到最底层来打印信息:

```c++
  // test/cpp/pir/cinn/pir_compiler_test.cc
  std::stringstream ss;
  program->Print(ss);
  LOG(INFO) << ss.str();
```


```c++
// paddle/pir/core/ir_printer.cc

void Program::Print(std::ostream& os) const {
  IrPrinter printer(os);
  printer.PrintProgram(this);
}
```


```c++
// paddle/pir/core/ir_printer.cc

void IrPrinter::PrintProgram(const Program* program) {
  auto top_level_op = program->module_op();
  for (size_t i = 0; i < top_level_op->num_regions(); ++i) {
    auto& region = top_level_op->region(i);
    PrintRegion(region);
  }
} 
```



```c++
// paddle/pir/core/ir_printer.cc

void IrPrinter::PrintRegion(const Region& region) {
  for (auto block : region) {
    PrintBlock(block);
  }
}

void IrPrinter::PrintBlock(const Block* block) {
  os << "{\n";
  for (auto item : *block) {
    PrintOperation(item);
    os << newline;
  }
  os << "}\n";
}

void IrPrinter::PrintOperation(Operation* op) {
  if (auto* dialect = op->dialect()) {
    dialect->PrintOperation(op, *this);
    return;
  }

  PrintGeneralOperation(op);
}
```


`IrPrinter` 内部定义了一系列方法, 来对 `Program`, `Region`, `Block`, `Operation`, `OpResult`, `AttributeMap` 和 `OpOperand` 实现打印操作, 会将所有信息都 `<<` 给传入的 `std::stringstream` 对象.


接下来到开始编译 `pir::Program` 到 `Runtime Program`, `Target target` 和 智能指针`std::shared_ptr<Scope> scope` 这俩个变量都是CINN中重要的数据结构. 

```c++
  // Step 2: Compiler New pir::Program into Runtime Program
  auto target = cinn::common::DefaultNVGPUTarget();
  auto scope = cinn::hlir::framework::BuildScope(target, *program);
  LOG(INFO) << scope->var_names().size();
  ASSERT_EQ(scope->var_names().size(), 8);

  cinn::hlir::framework::PIRCompiler ir_compiler(*program, target, scope);
  auto runtime_program = ir_compiler.Build(groups);
```


`Target` 描述了当前环境的操作系统内容, 其有 5 个枚举类 `OS`, `Arch`, `Bit`, `Feature` 和 `Lib` 分别描述了操作系统的特点.


```c++
struct Target {

  enum class OS : int {
    Unk = -1,
    Linux,
    Windows,
  };

  /**
   * The architecture used by the target. Determines the instruction set to use.
   */
  enum class Arch : int {
    Unk = -1,
    X86,
    ARM,
    NVGPU,
  };

  enum class Bit : int {
    Unk = -1,
    k32,
    k64,
  };

  OS os{OS::Unk};
  Arch arch{Arch::Unk};
  Bit bits{Bit::Unk};

  enum class Feature : int {
    JIT = 0,
    Debug,
  };

  /**
   * The library used by the target.
   */
  enum class Lib : int {
    Unk = -1,
    MKL,
  };
```

在当前单测中, 我们只需要看这两个函数

成员函数 `defined` 描述了当前 `target` 是否已经初始化过, 而 `DefaultNVGPUTarget` 用来返回一个NVGPU的target.

```c++
  // paddle\cinn\common\target.h

  bool defined() const {
    return os != OS::Unk && arch != Arch::Unk && bits != Bit::Unk;
  }
```

```c++
// paddle/cinn/common/target.cc
const Target &DefaultNVGPUTarget() {
  static Target target(
      Target::OS::Linux, Target::Arch::NVGPU, Target::Bit::k64, {}, {});
  return target;
}
```


`Target` 相对来说简单一些, `Scope` 相对复杂一些, 不过我们依旧抓住重点就可以, 它有一个私有变量 `data_`, 类型是 `absl::flat_hash_map`, 我们当做一个 `unordered_map` 就好, `Scope` 就是在该 map 中, 注册了一系列的 `Variable` 指针, 而 `Variable` 我们可以暂时理解为 `Tensor`.

```c++
class Scope {
 public:

  // ......

  //! Get or create a variable.
  template <typename T>
  Variable* Var(const std::string& name);

 private:
  absl::flat_hash_map<std::string, std::unique_ptr<Variable>> data_;

  // ......
};
```

成员模板函数 `Scope::Var` 参数传入对应的 `name` 字符串, 如果已经注册过, 则直接返回之前的结果, 否则根据模板参数, 来创建一个空的 `Variable` 并注册.

```c++
template <typename T>
Variable* Scope::Var(const std::string& name) {
  VLOG(4) << "Scope insert Var [" << name << "]";
  Variable* x = FindVar(name);
  if (x) return x;
  auto* data = new Variable(T());
  data_[name].reset(data);
  return data;
}
```




































