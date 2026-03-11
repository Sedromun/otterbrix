#include "create_plan_vector_search.hpp"

#include <components/expressions/compare_expression.hpp>
#include <components/logical_plan/node_vector_search.hpp>
#include <components/physical_plan/operators/operator_vector_search.hpp>

namespace services::planner::impl {

    components::operators::operator_ptr create_plan_vector_search(const context_storage_t& context,
                                                                  const components::logical_plan::node_ptr& node) {
        auto vs_node = static_cast<const components::logical_plan::node_vector_search_t*>(node.get());

        // Extract optional filter expression (stored as first expression on the node)
        components::expressions::compare_expression_ptr filter;
        if (!node->expressions().empty()) {
            filter = reinterpret_cast<const components::expressions::compare_expression_ptr&>(node->expressions()[0]);
        }

        if (context.has_collection(node->collection_full_name())) {
            return boost::intrusive_ptr(
                new components::operators::operator_vector_search_t(context.resource,
                                                                    context.log.clone(),
                                                                    node->collection_full_name(),
                                                                    vs_node->column_name(),
                                                                    vs_node->query_vector(),
                                                                    vs_node->k(),
                                                                    vs_node->metric(),
                                                                    filter));
        } else {
            return boost::intrusive_ptr(
                new components::operators::operator_vector_search_t(nullptr,
                                                                    log_t{},
                                                                    node->collection_full_name(),
                                                                    vs_node->column_name(),
                                                                    vs_node->query_vector(),
                                                                    vs_node->k(),
                                                                    vs_node->metric(),
                                                                    filter));
        }
    }

} // namespace services::planner::impl
