defmodule GPotion.TypeCheck do
  @allowed_types_var ~w(int float)a
  @allowed_types_kernel ~w(gmatrex integer float unit)a
  @allowed_constants ~w(blockIdx blockDim gridDim threadIdx)a

  def init_check({:__block__, _, [{:gptype, _, [{kernel_name, _, list_types}]},{:gpotion, _,[{kernel_name, _, list_var}, body]}]}) do
    map = create_map(list_types, list_var)
    check_types(map, body)
  end

  defp create_map(list_types, list_var) do
    types_with_int = Enum.map(extract_types(list_types), &adjust_integer_type/1)
    Enum.zip(extract_var(list_var), types_with_int)
    |> Enum.into(%{})
  end

  def extract_var([]), do: []

  def extract_var([{name, _, _} | rest]), do: [name | extract_var(rest)]

  def extract_types([{:~>, _, children}]) do
    extract_types(children)
  end

  def extract_types([]), do: []

  def extract_types([{type, _, _} | rest]) when type in @allowed_types_kernel do
    [type | extract_types(rest)]
  end

  def extract_types([{type, _, _} | rest]) when type in @allowed_types_kernel do
    [type | extract_types(rest)]
  end

  def extract_types([{_, _, children} | rest]), do: extract_types(children) ++ extract_types(rest)

  def extract_types_tuple([], acc), do: acc

  def extract_types_tuple([{type, _, _} | rest], acc) when type == :~> do
    extract_types_tuple(rest, acc)
  end

  def extract_types_tuple([{type, _, _} | rest], acc) when type in @allowed_types_kernel do
    extract_types_tuple(rest, [type | acc])
  end

  def extract_types_tuple([{_, _, children} | rest], acc), do: extract_types_tuple(children ++ rest, acc)


  def check_types(map, body) do
    case body do
      {:__block__, _, code} ->
        check_block(map, code)
      {:do, {:__block__,_, code}} ->
        check_block(map, code)
      {:do, code} ->
        check_command(map, code)
      [do: code] ->
        check_types(map, code)
      {_,_,_} ->
        check_command(map, body)
    end
  end

  defp check_block(map, code) do
    Enum.reduce(code, map, fn com, acc -> check_types(acc,com) end)
  end

  #checagem de comandos
  def check_command(map, code) do

    case code do

      #Declaração com atribuição
      {:var, _, [{var, _, [{:=, _, [{type, _, _}, exp]}]}]} ->
        exp_type = check_exp(map, exp)
        if type != exp_type do
          raise "Error: Incompatible type for variable '#{var}'. Expected '#{type}' but received '#{exp_type}'"
        else
          check_var(map, var, type)
        end

      # Declaração de variável sem atribuição
      {:var, _, [{var, _, [{type, _, _}]}]} ->
        check_var(map, var, type)

      # Atribuição de vetor
      {:=, _, [{{:., _, _}, _, [name, index]}, value]} ->
        type_arr = check_exp(map, name)
        if type_arr != :gmatrex do
          raise "Error: Unexpected type. Expected 'gmatrex', but received '#{type_arr}'."
        end
        type_value = check_exp(map, value)
        if type_value != :float do
          raise "Error: Incompatible type for variable. Expected 'float' but received '#{type_value}'."
        end
        index_type = check_exp(map, index)
        if index_type != :int do
          raise "Error: Invalid index type for array access. Array indices must be integers, but a non-integer value was provided."
        end
        map

      # Atribuição
      {:=, _, [{var, _, _}, exp]} ->
        var_type = Map.get(map, var)
        exp_type = check_exp(map, exp)
        if var_type != exp_type do
          raise "Error: Incompatible type for variable '#{var}'. Expected '#{var_type}' but received '#{exp_type}'"
        else
          map
        end

      # Sequencia de comandos
      [c1, c2] ->
        new_map = check_types(map, c1)
        check_types(new_map, c2)

      # IF ELSE
      {:if, _,[cond,[do: b1, else: b2]]} ->
        cond_type = check_exp(map, cond)
        if cond_type != :int do
          raise "Error: Incompatible conditional expression type. Expected 'int' but received '#{cond_type}'"
        else
          check_types(map, b1)
          check_types(map, b2)
          map
        end

      # IF
      {:if, _,[cond, [do: body]]} ->
        cond_type = check_exp(map, cond)
        if cond_type != :int do
          raise "Error: Incompatible conditional expression type. Expected 'int' but received '#{cond_type}'"
        else
          check_types(map, body)
          map
        end

      # FOR
      {:for, _,[{:in, _,[{var, _, _}, {:range, _, [value1, value2]}]},[do: body]]} ->
      if check_exp(map, value1) != :int or check_exp(map, value2) != :int do
        raise "Error: Incompatible types in loop range. Expected integer values for the loop range, but received non-integer values."
      else
        check_types(Map.put(map, var, :int), body)
        map
      end

      #WHILE
      {:while, _, [cond, [do: body]]} ->
        cond_type = check_exp(map, cond)
        if cond_type != :int do
          raise "Error: Incompatible conditional expression type. Expected 'int' but received '#{cond_type}'"
        else
          check_types(map, body)
          map
        end

      _ -> map

    end

  end

  #checagem de expressões
  def check_exp(map, code) do

    case code do

      # operadores artmeticos
      {op, _, args} when op in [:+, :-, :/, :*] ->
        case args do

          [e1] ->
            check_exp(map, e1)

          [e1,e2] ->
            t1 = check_exp(map, e1)
            t2 = check_exp(map, e2)
            case t1 do
              :int  -> case t2 do
                         :int -> :int
                         :float -> :float
                        end
              :float -> :float

            end
        end

      # Operadores relacionais
      {op, _, args} when op in [:<=, :<, :>, :>=,:!=,:==] ->
        case args do

          [e1] ->
            check_exp(map, e1)

          [e1,e2] ->
            check_exp(map, e1)
            check_exp(map, e2)
            :int
      end

      # Operadores lógicos
      {op, _, args} when op in [:&&, :||, :!] ->
        case args do

          [e1] ->
            t1 = check_exp(map, e1)
            if t1 != :int do
              raise "Error: Incompatible expression type. Expected 'int' but received '#{t1}'"
            end
            :int

          [e1,e2] ->
            t1 = check_exp(map, e1)
            t2 = check_exp(map, e2)
            if t1 != :int or t2 != :int do
              raise "Error: Incompatible expression type. Expected 'int' but received '#{t1}' and '#{t2}'"
            end
            :int
        end

      # Constantes
      {{:., _, [{const, _, _}, _]}, _, _} when const in @allowed_constants -> :int

      # Vetores
      {{:., _, _}, _, [name, index]} ->
        type_arr = check_exp(map, name)
        if type_arr != :gmatrex do
          raise "Error: Unexpected type. Expected 'gmatrex', but received '#{type_arr}'."
        end
        if check_exp(map, index) != :int do
          raise "Error: Invalid index type for array access. Array indices must be integers, but a non-integer value was provided."
        end
        :float

      # Variaveis
      {var, _, _} ->
        type = Map.get(map, var)
        if type == nil do
          raise "Error: variable '#{var}' is used in an expression before being declared."
        else
          type
        end

      # Valores literais
      float when  is_float(float) -> :float
      int   when  is_integer(int) -> :int

      _ -> :int
    end

  end

  # utils
  def check_var(map, var, type) do
    if Map.get(map, var) do
      raise "Error: Variable '#{var}' is already defined."
    else
      if type in @allowed_types_var do
        Map.put(map, var, type)
      else
        raise "Error: Type '#{type}' is not allowed."
      end
    end
  end

  defp adjust_integer_type(:integer), do: :int
  defp adjust_integer_type(type), do: type

end
