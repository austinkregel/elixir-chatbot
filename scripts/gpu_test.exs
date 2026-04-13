IO.puts("=== GPU TEST START ===")
IO.puts("Default backend: #{inspect(Nx.default_backend())}")

t = Nx.tensor([1.0, 2.0, 3.0])
IO.puts("Tensor created on backend: #{inspect(t.data.__struct__)}")

fun = Nx.Defn.jit(fn x -> Nx.add(x, 1) end, compiler: EXLA)
IO.puts("JIT function compiled")

result = fun.(t)
IO.puts("Result: #{inspect(result)}")
IO.puts("=== GPU TEST PASS ===")
