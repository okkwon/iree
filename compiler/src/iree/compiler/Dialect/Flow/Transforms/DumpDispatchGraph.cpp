// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===------------------- DumpDispatchGraph.cpp ----------------------------===//
//
// Generate a graphviz graph for dispatches
//
//===----------------------------------------------------------------------===//

#include <utility>

#include "PassDetail.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/GraphWriter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/IndentedOstream.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

namespace {

static const StringRef kLineStyleControlFlow = "dashed";
static const StringRef kLineStyleDataFlow = "solid";
static const StringRef kShapeNode = "box";
static const StringRef kShapeBox = "box";
static const StringRef kShapeTab = "tab";
static const StringRef kShapeNone = "plain";
static const StringRef kShapeEllipse = "ellipse";

static StringRef getShape(Operation *op) {
  if (isa<DispatchOp>(op)) return kShapeBox;

  return kShapeEllipse;
}

/// Return the size limits for eliding large attributes.
static int64_t getLargeAttributeSizeLimit() {
  // Use the default from the printer flags if possible.
  if (Optional<int64_t> limit = OpPrintingFlags().getLargeElementsAttrLimit())
    return *limit;
  return 16;
}

/// Return all values printed onto a stream as a string.
static std::string strFromOs(function_ref<void(raw_ostream &)> func) {
  std::string buf;
  llvm::raw_string_ostream os(buf);
  func(os);
  return os.str();
}

/// Escape special characters such as '\n' and quotation marks.
static std::string escapeString(std::string str) {
  return strFromOs([&](raw_ostream &os) {
    for (unsigned char c : str) {
      switch (c) {
        case '\\':
          os << '\\' << '\\';
          break;
        case '\t':
          os << '\\' << 't';
          break;
        case '\n':
          os << '\\' << 'n';
          break;
        case '"':
          os << '\\' << '"';
          break;
        case '\r':  // translate "carriage return" as "\l"
          os << '\\' << 'n';
          break;
        default:
          if (llvm::isPrint(c)) {
            os << c;
            break;
          }

          // Always use a full 3-character octal escape.
          os << '\\';
          os << char('0' + ((c >> 6) & 7));
          os << char('0' + ((c >> 3) & 7));
          os << char('0' + ((c >> 0) & 7));
      }
    }
  });
}

/// Put quotation marks around a given string.
static std::string quoteString(const std::string &str) {
  return "\"" + str + "\"";
}

using AttributeMap = llvm::StringMap<std::string>;

/// This struct represents a node in the DOT language. Each node has an
/// identifier and an optional identifier for the cluster (subgraph) that
/// contains the node.
/// Note: In the DOT language, edges can be drawn only from nodes to nodes, but
/// not between clusters. However, edges can be clipped to the boundary of a
/// cluster with `lhead` and `ltail` attributes. Therefore, when creating a new
/// cluster, an invisible "anchor" node is created.
struct Node {
 public:
  Node(int id = 0, Optional<int> clusterId = llvm::None)
      : id(id), clusterId(clusterId) {}

  Node(int id, std::string label, StringRef shape,
       Optional<int> clusterId = llvm::None)
      : id(id), label(label), shape(shape), clusterId(clusterId) {}

  int id;
  std::string label;
  StringRef shape;
  Optional<int> clusterId;
};

/// This pass generates a Graphviz dataflow visualization of an MLIR operation.
/// Note: See https://www.graphviz.org/doc/info/lang.html for more information
/// about the Graphviz DOT language.
class DumpDispatchGraphPass
    : public DumpDispatchGraphBase<DumpDispatchGraphPass> {
 public:
  DumpDispatchGraphPass(raw_ostream &os) : os(os) {}
  DumpDispatchGraphPass(const DumpDispatchGraphPass &o)
      : DumpDispatchGraphPass(o.os.getOStream()) {}

  void runOnOperation() override {
    auto modOp = dyn_cast<ModuleOp>(getOperation());
    if (!modOp) return;

    SmallVector<func::FuncOp> funcOps(modOp.getOps<func::FuncOp>());

    if (funcOps.empty() || funcOps.size() > 2) return;
    func::FuncOp entryFunc;

    if (funcOps.size() == 1)
      entryFunc = funcOps[0];
    else {
      // When there are two func ops, the public one is the wrapper.
      entryFunc = funcOps[0].isPublic() ? funcOps[1] : funcOps[0];
    }
    // emitGraphVisJS([&]() {
    //   for (auto funcOp : funcOps) processOperation(funcOp);
    //   emitAllVisJS();
    // });

    emitGraphCytoscape([&]() {
      processOperation(entryFunc);
      emitCytoscapeData();
    });

    for (Node *node : nodes) {
      delete node;
    }
  }

 private:
  /// Emit a cluster (subgraph). The specified builder generates the body of the
  /// cluster. Return the anchor node of the cluster.
  void emitClusterStmt(function_ref<void()> builder, std::string label = "") {
    ++clusterId;
    // os << "subgraph cluster_" << clusterId << " {\n";
    // os.indent();
    // Emit invisible anchor node from/to which arrows can be drawn.
    // Node anchorNode = emitNodeStmt(" ", kShapeNone);
    // os << attrStmt("label", quoteString(escapeString(std::move(label))))
    //    << ";\n";
    builder();
    // os.unindent();
    // os << "}\n";
  }

  /// Emit a cluster (subgraph). The specified builder generates the body of the
  /// cluster. Return the anchor node of the cluster.
  void visitRegion(function_ref<void()> builder, std::string label = "") {
    builder();
  }

  /// Generate an attribute statement.
  std::string attrStmt(const Twine &key, const Twine &value) {
    return (key + " = " + value).str();
  }

  /// Emit an attribute list.
  void emitAttrList(raw_ostream &os, const AttributeMap &map) {
    os << "[";
    interleaveComma(map, os, [&](const auto &it) {
      os << this->attrStmt(it.getKey(), it.getValue());
    });
    os << "]";
  }

  // Print an MLIR attribute to `os`. Large attributes are truncated.
  void emitMlirAttr(raw_ostream &os, Attribute attr) {
    // A value used to elide large container attribute.
    int64_t largeAttrLimit = getLargeAttributeSizeLimit();

    // Always emit splat attributes.
    if (attr.isa<SplatElementsAttr>()) {
      attr.print(os);
      return;
    }

    // Elide "big" elements attributes.
    auto elements = attr.dyn_cast<ElementsAttr>();
    if (elements && elements.getNumElements() > largeAttrLimit) {
      os << std::string(elements.getType().getRank(), '[') << "..."
         << std::string(elements.getType().getRank(), ']') << " : "
         << elements.getType();
      return;
    }

    auto array = attr.dyn_cast<ArrayAttr>();
    if (array && static_cast<int64_t>(array.size()) > largeAttrLimit) {
      os << "[...]";
      return;
    }

    // Print all other attributes.
    std::string buf;
    llvm::raw_string_ostream ss(buf);
    attr.print(ss);
    os << truncateString(ss.str());
  }

  /// Append an edge to the list of edges.
  /// Note: Edges are written to the output stream via `emitAllEdgeStmts`.
  void emitEdgeStmt(Node n1, Node n2, std::string label, StringRef style) {
    AttributeMap attrs;
    attrs["style"] = style.str();
    // Do not label edges that start/end at a cluster boundary. Such edges are
    // clipped at the boundary, but labels are not. This can lead to labels
    // floating around without any edge next to them.
    if (!n1.clusterId && !n2.clusterId)
      attrs["label"] = quoteString(escapeString(std::move(label)));
    // Use `ltail` and `lhead` to draw edges between clusters.
    if (n1.clusterId)
      attrs["ltail"] = "cluster_" + std::to_string(*n1.clusterId);
    if (n2.clusterId)
      attrs["lhead"] = "cluster_" + std::to_string(*n2.clusterId);

    edges.push_back(strFromOs([&](raw_ostream &os) {
      os << llvm::format("v%i -> v%i ", n1.id, n2.id);
      emitAttrList(os, attrs);
    }));
  }

  /// Emit a graph. The specified builder generates the body of the graph.
  void emitGraphVisJS(function_ref<void()> builder) {
    os << "<html>\n"
       << "<head>\n"
       << "  <script type=\"text/javascript\" "
          "src=\"https://unpkg.com/vis-network/standalone/umd/"
          "vis-network.min.js\"></script>\n"
       << "</head>\n"
       << "<body>\n"
       << "<div id=\"mynetwork\"></div>\n"
       << "<script type=\"text/javascript\">\n"
       << "  var options = {};\n"
       << "  var container = document.getElementById('mynetwork');\n";
    builder();
    os << "  var data = {\n"
       << "    nodes: nodes,\n"
       << "    edges: edges\n"
       << "  };\n"
       << "  var network = new vis.Network(container, data, options);\n"
       << "</script>\n"
       << "</body>\n"
       << "</html>\n";
  }

  void emitAllVisJS() {
    os << "  var nodes = new vis.DataSet([\n";
    for (size_t i = 0, e = nodes.size(); i != e; ++i) {
      Node *node = nodes[i];
      os << "    { id: " << node->id << ", label: " << node->label << " }";
      if (i == e - 1)
        os << "\n";
      else
        os << ",\n";
    }
    os << "]);\n";

    os << "  var edges = new vis.DataSet([\n";

    SmallVector<std::pair<int64_t, int64_t>> edges;

    for (Operation *op : visitedOperations) {
      auto toNode = operationToNode[op];

      // Insert data flow edges originating from each operand.
      unsigned numOperands = op->getNumOperands();
      for (unsigned i = 0; i < numOperands; i++) {
        auto operand = op->getOperand(i);

        // a constant operand is not going to be available in the map.
        if (valueToNode.count(operand)) {
          Node *fromNode = valueToNode[operand];
          edges.push_back({fromNode->id, toNode->id});
        }
      }
    }
    for (size_t i = 0, e = edges.size(); i != e; ++i) {
      std::pair<int64_t, int64_t> edge = edges[i];
      const int64_t from = std::get<0>(edge);
      const int64_t to = std::get<1>(edge);
      os << "    {from: " << from << ", to: " << to << "}";
      if (i == e - 1)
        os << "\n";
      else
        os << ",\n";
    }
    os << "  ]);\n";
  }

  /// Emit a graph. The specified builder generates the body of the graph.
  void emitGraphCytoscape(function_ref<void()> builder) { builder(); }

  void emitCytoscapeData() {
    os << "elements: {\n"
       << "  nodes: [\n";
    for (size_t i = 0, e = nodes.size(); i != e; ++i) {
      Node *node = nodes[i];
      os << "    { data: { id: '" << node->id << "', label: " << node->label
         << " } }";
      if (i == e - 1)
        os << "\n";
      else
        os << ",\n";
    }
    os << "  ],\n"
       << "  edges: [\n";

    SmallVector<std::pair<int64_t, int64_t>> edges;

    // A same value can appear multiple times as operand
    DenseMap<Value, bool> visited;

    for (Operation *op : visitedOperations) {
      auto toNode = operationToNode[op];

      // Insert data flow edges originating from each operand.

      visited.clear();

      unsigned numOperands = op->getNumOperands();
      for (unsigned i = 0; i < numOperands; i++) {
        Value operand = op->getOperand(i);
        if (visited.count(operand)) continue;
        visited[operand] = true;
        // a constant operand is not going to be available in the map.
        if (valueToNode.count(operand)) {
          Node *fromNode = valueToNode[operand];
          edges.push_back({fromNode->id, toNode->id});
        }
      }
    }
    for (size_t i = 0, e = edges.size(); i != e; ++i) {
      std::pair<int64_t, int64_t> edge = edges[i];
      const int64_t from = std::get<0>(edge);
      const int64_t to = std::get<1>(edge);
      os << "    { data: { source: '" << from << "', target: '" << to
         << "' } }";
      if (i == e - 1)
        os << "\n";
      else
        os << ",\n";
    }
    os << "  ]\n"
       << "}\n";
  }

  /// Emit a node statement.
  void visitOperationToCreateNode(Operation *op) {
    ++nodeId;
    auto shape = getShape(op);
    auto label = quoteString(escapeString(getLabel(op)));
    auto node = new Node(nodeId, label, shape);
    nodes.push_back(node);
    operationToNode[op] = node;
    for (Value result : op->getResults()) valueToNode[result] = node;
    visitedOperations.push_back(op);
  }

  void visitBlockArg(BlockArgument &blockArg) {
    ++nodeId;
    auto label = quoteString(escapeString(getLabel(blockArg)));
    auto node = new Node(nodeId, label, kShapeNode);
    nodes.push_back(node);
    valueToNode[blockArg] = node;
  }

  void printResults(raw_ostream &os, Operation *op, AsmState &state) {
    for (auto result : op->getResults()) {
      result.printAsOperand(os, state);
    }
  }

  void printResultsAndName(raw_ostream &os, Operation *op, AsmState &state) {
    printResults(os, op, state);
    os << " = " << op->getName();
  }

  void printDispatchTensorLoad(raw_ostream &os, DispatchTensorLoadOp op,
                               AsmState &state) {
    printResultsAndName(os, op.getOperation(), state);
    os << " ";
    op.getSource().printAsOperand(os, state);
    os << " -> " << op.getResult().getType();
    os << "\r";
  }

  void printDispatchTensorStore(raw_ostream &os, DispatchTensorStoreOp op,
                                AsmState &state) {
    os << op->getName() << " ";
    op.getValue().printAsOperand(os, state);
    os << ", ";
    op.getTarget().printAsOperand(os, state);
    os << "\r";
  }

  void printGeneric(raw_ostream &os, linalg::GenericOp op, AsmState &state) {
    printLinalgInsOuts(os, op, state);
    for (Operation &operation : *op.getBlock()) {
      os.indent(8);
      annotateOperation(os, &operation, state);
    }
  }

  template <typename T>
  void printLinalgInsOuts(raw_ostream &os, T op, AsmState &state) {
    printResultsAndName(os, op.getOperation(), state);
    os << " " << op.iterator_types() << "(";
    printOperands(os, op.getInputs(), state);
    os << ") -> (";
    printOperands(os, op.getOutputs(), state);
    os << ")\r";
  }

  void annotateOperation(raw_ostream &os, Operation *op, AsmState &state) {
    if (isa<arith::ConstantOp>(op)) return;

    if (isa<func::ReturnOp>(op)) return;

    if (auto load = dyn_cast<DispatchTensorLoadOp>(op)) {
      printDispatchTensorLoad(os, load, state);
      return;
    }

    if (auto store = dyn_cast<DispatchTensorStoreOp>(op)) {
      printDispatchTensorStore(os, store, state);
      return;
    }

    if (auto generic = dyn_cast<linalg::GenericOp>(op)) {
      printGeneric(os, generic, state);
      return;
    }

    if (auto linalgOp = dyn_cast<linalg::MatmulOp>(op)) {
      printLinalgInsOuts(os, linalgOp, state);
      return;
    }

    if (auto linalgOp = dyn_cast<linalg::BatchMatmulOp>(op)) {
      printLinalgInsOuts(os, linalgOp, state);
      return;
    }

    os << *op << "\r";
  }

  void printDispatchBody(raw_ostream &os, DispatchOp &dispatchOp) {
    // Find the entry point function from the dispatch entry point symbol
    // attribute.
    auto entryPoint = dispatchOp.getEntryPoint();
    auto executableOp = cast<ExecutableOp>(SymbolTable::lookupNearestSymbolFrom(
        dispatchOp, entryPoint.getRootReference()));
    if (!executableOp) return;

    auto calleeNameAttr = entryPoint.getLeafReference();
    auto innerModule = executableOp.getInnerModule();
    auto funcOps = innerModule.getOps<func::FuncOp>();
    auto funcIt = llvm::find_if(funcOps, [&](func::FuncOp op) {
      return op.getNameAttr() == calleeNameAttr;
    });
    if (funcIt == funcOps.end()) return;

    auto callee = *funcIt;

    AsmState state(callee);

    // Iterate the operations of the function body and print important
    // operation.
    for (auto &block : callee.getBlocks()) {
      for (auto &op : block.getOperations()) {
        annotateOperation(os, &op, state);
      }
    }
  }

  void printOperands(raw_ostream &os, ::mlir::Operation::operand_range operands,
                     AsmState &state) {
    auto numOperands = operands.size();

    for (auto it : llvm::enumerate(operands)) {
      auto operand = it.value();
      auto op = operand.getDefiningOp();

      if (op && isScalarConstantOp(op)) {
        auto ty = operand.getType();
        if (ty.isa<IntegerType>()) {
          os << cast<arith::ConstantIntOp>(op).value();
        } else if (ty.isa<FloatType>()) {
          cast<arith::ConstantFloatOp>(op).value().print(os);
        } else {
          os << cast<arith::ConstantIndexOp>(op).value();
        }
      } else {
        operand.printAsOperand(os, state);
      }

      if (it.index() != numOperands - 1) {
        os << ", ";
      }
    }
  }

  /// Generate a label for an operation.
  std::string getLabel(Operation *op) {
    return strFromOs([&](raw_ostream &os) {
      if (op->getNumRegions() == 0) {
        auto funcOp = op->getParentOfType<func::FuncOp>();
        AsmState state(funcOp);
        printResults(os, op, state);
        os << " = " << op->getName();

        if (auto dispatch = dyn_cast<DispatchOp>(op)) {
          // print workload
          os << "[";
          printOperands(os, dispatch.getWorkload(), state);
          os << "]\n";

          // Print entry function name, if there is only one entry function,
          // then the name space and the entry function names are the same,
          // and we can just print the function name to save space.
          auto entryPoint = dispatch.getEntryPoint();
          auto rootName = entryPoint.getRootReference();
          auto leafName = entryPoint.getLeafReference();
          if (rootName == leafName) {
            os << leafName;
          } else {
            os << entryPoint;  // print the full name
          }

          // print entry function args
          os << "(";
          printOperands(os, dispatch.getArguments(), state);
          os << ")\n";

          printDispatchBody(os, dispatch);

        } else {
          os << "\n";
        }
      } else {
        os << op->getName() << "\n";
      }

      if (printResultTypes) {
        std::string buf;
        llvm::raw_string_ostream ss(buf);
        interleave(op->getResultTypes(), ss, "\n");
        os << ss.str();
      }
    });
  }

  /// Generate a label for a block argument.
  std::string getLabel(BlockArgument arg) {
    return "arg" + std::to_string(arg.getArgNumber());
  }

  /// Process a block. Emit a cluster and one node per block argument and
  /// operation inside the cluster.
  void processBlock(Block &block) {
    emitClusterStmt([&]() {
      for (BlockArgument &blockArg : block.getArguments())
        visitBlockArg(blockArg);

      for (Operation &op : block) {
        processOperation(&op);
      }
    });
  }

  bool isScalarConstantOp(Operation *op) {
    if (auto constOp = dyn_cast<mlir::arith::ConstantOp>(op))
      if (constOp.getResult().getType().isIntOrIndexOrFloat()) return true;

    return false;
  }

  /// Process an operation. If the operation has regions, emit a cluster.
  /// Otherwise, emit a node.
  void processOperation(Operation *op) {
    // // Do not handle some noisy Operations.
    if (isa<arith::ConstantOp>(op) || isa<Util::GlobalLoadOpInterface>(op)) {
      return;
    }

    // skip hal.buffer_view.dim
    // if (isa<IREE::HAL::BufferViewDimOp>(op)) return;

    // if (isa<AffineApplyOp>(op)) return;

    // if (isa<arith::ArithmeticDialect>(op->getDialect())) return;

    if (op->getNumRegions() == 1) {
      // do not generate a cluster when there is one region.
      processRegion(op->getRegion(0));
    } else if (op->getNumRegions() > 1) {
      // Emit cluster for op with regions.
      visitRegion(
          [&]() {
            for (Region &region : op->getRegions()) processRegion(region);
          },
          getLabel(op));
    } else {
      visitOperationToCreateNode(op);
    }

    return;
  }

  /// Process a region.
  void processRegion(Region &region) {
    for (Block &block : region.getBlocks()) processBlock(block);
  }

  /// Truncate long strings.
  std::string truncateString(std::string str) {
    if (str.length() <= maxLabelLen) return str;
    return str.substr(0, maxLabelLen) + "...";
  }

  /// Output stream to write DOT file to.
  raw_indented_ostream os;
  /// A list of edges. For simplicity, should be emitted after all nodes were
  /// emitted.
  std::vector<std::string> edges;
  /// Mapping of SSA values to graph nodes/clusters.
  DenseMap<Value, Node *> valueToNode;
  /// Mapping of Operation * to graph nodes
  DenseMap<Operation *, Node *> operationToNode;

  /// Counter for generating unique node/subgraph identifiers.
  int nodeId = 0;
  int clusterId = 0;

  SmallVector<Operation *> visitedOperations;
  SmallVector<Value> visitedValue;
  SmallVector<Node *> nodes;
};

}  // namespace

std::unique_ptr<Pass> createDumpDispatchGraphPass(raw_ostream &os) {
  return std::make_unique<DumpDispatchGraphPass>(os);
}

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
