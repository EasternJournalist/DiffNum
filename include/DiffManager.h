#pragma once

#include <DiffMath.h>
#include <unordered_map>

namespace DiffNum {
	/// <summary>
	/// Manager for variables. Add all independent variables to the manager and let it set up the gradients for you.
	/// Otherwise you may use DiffVar::setVar to set up manually.
	/// </summary>
	/// <typeparam name="n_type"></typeparam>
	template <typename n_type>
	class DiffManager {
	public:
		DiffManager() : not_done(true) {}
		~DiffManager() {}

		/// <summary>
		/// Add one independent variable to manager.
		/// </summary>
		/// <param name="_Var"></param>
		/// <returns> The index of the variable in gradient array. </returns>
		ptrdiff_t AddVariable(DiffVar<n_type, 0>* _Var) {
			indices[_Var] = variables.size();
			variables.push_back(_Var);
			return variables.size() - 1;
		}

		/// <summary>
		/// Add an array of independent variables to manager.
		/// </summary>
		/// <param name="_Begin">Pointer to the first variable to be added </param>
		/// <param name="_Size">Number of variables to be added </param>
		/// <returns>The index of the first variable </returns>
		ptrdiff_t AddVariables(DiffVar<n_type, 0>* _Begin, ptrdiff_t _Size) {
			for (DiffVar<n_type, 0>* _var = _Begin; _var < _Begin + _Size; _var++) {
				indices[_var] = variables.size();
				variables.push_back(_var);
			}
			return variables.size() - _Size;
		}

		/// <summary>
		/// Set up variable gradients.
		/// </summary>
		void SetUp() {
			ptrdiff_t num_var = variables.size();
			for (ptrdiff_t i = 0; i < num_var; i++) {
				variables[i]->setVar(num_var, i);
			}
			not_done = false;
		}

		/// <summary>
		/// Get the index of _Var in gradient array
		/// </summary>
		/// <param name="_Var">Pointer to the variable. </param>
		/// <returns></returns>
		ptrdiff_t Index(DiffVar<n_type, 0>* _Var) const {
			return indices[_Var];
		}
		
		/// <summary>
		/// Get the index of _Var in gradient array
		/// </summary>
		/// <param name="_Var">const reference to the variable. </param>
		/// <returns></returns>
		ptrdiff_t Index(const DiffVar<n_type, 0>& _Var) const {
			return indices[&_Var];
		}

	private:
		bool not_done;

		std::vector<DiffVar<n_type, 0>*> variables;
		std::unordered_map<DiffVar<n_type, 0>*, ptrdiff_t> indices;
	};


}