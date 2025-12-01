import torch
import torch.nn as nn


class ResNetEuler(nn.Module):
    """
    Implementación de una ResNet profunda vista como una discretización de Euler
    de una EDO controlada.

    Dinámica: y_{j+1} = y_j + delta_t * activation(K_j * y_j + beta_j)
    Entrenamiento: Basado en el Principio del Máximo de Pontryagin (PMP).
    """

    def __init__(
        self,
        dim: int,
        num_layers: int,
        activation,
        activation_derivative,
        hip_function,
        hip_function_derivative,
        delta_t: float = 1.0,
    ) -> None:
        super().__init__()
        self.N = num_layers
        self.delta_t = delta_t  # Paso de tiempo para Euler

        # Capas internas: representan los controles K(t) y beta(t) discretizados
        self.inner_layers = nn.ModuleList(
            [nn.Linear(dim, dim) for _ in range(num_layers)]
        )

        # Capa final de proyección para clasificación
        self.last_layer = nn.Linear(dim, 1)

        # Funciones de activación y derivadas (necesarias para backprop manual)
        self.activation = activation
        self.activation_derivative = activation_derivative

        # Función de hipótesis final (ej. Sigmoide para probabilidad)
        self.hip_function = hip_function
        self.hip_function_derivative = hip_function_derivative

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """
        Paso forward estándar (integración de la EDO).
        """
        for layer in self.inner_layers:
            a = layer(y)  # a = K*y + beta
            y = y + self.delta_t * self.activation(a)  # Euler step
        z = self.last_layer(y)
        return self.hip_function(z)

    @torch.no_grad()
    def calculate_y_trayectory(self, y0: torch.Tensor):
        """
        Calcula y almacena toda la trayectoria del estado 'y' y las pre-activaciones 'a'.
        Necesario para el paso backward (cálculo de p).
        """
        y_traj = [y0]
        a_list = []
        y = y0
        for layer in self.inner_layers:
            a = layer(y)
            a_list.append(a)
            # Actualización de estado tipo Euler
            y = y + self.delta_t * self.activation(a)
            y_traj.append(y)

        y_traj = torch.stack(y_traj, dim=0)  # Shape: (N_layers+1, Batch, Dim)
        a_list = torch.stack(a_list, dim=0)  # Shape: (N_layers, Batch, Dim)
        return y_traj, a_list

    @torch.no_grad()
    def calculate_p_trayectory(self, y_traj: torch.Tensor, a_list, c: torch.Tensor):
        """
        Calcula la trayectoria del estado adjunto (costate) 'p' retrocediendo en el tiempo.
        Esto corresponde a resolver la Ecuación Adjunta discretizada.

        p_N = gradiente de la función de costo terminal respecto a y_N.
        p_j = p_{j+1} + delta_t * (transpuesta(Jacobiano_f) * p_{j+1})
        """
        # 1. Condición final para p (Transversality condition)
        yN = y_traj[-1]
        z = self.last_layer(yN)
        C = self.hip_function(z)  # Predicción
        Cp = self.hip_function_derivative(z)

        c = c.view(-1, 1)  # Etiquetas reales

        # epsilon es el error en la capa final escalado por la derivada de hip
        epsilon = (C - c) * Cp
        W = self.last_layer.weight

        # p_N inicial
        p = epsilon @ W

        p_list = [None] * (self.N + 1)
        p_list[self.N] = p

        # 2. Iteración hacia atrás (Backward pass manual)
        for j in reversed(range(self.N)):
            layer = self.inner_layers[j]
            a_j = a_list[j]

            # Derivada de sigma respecto al argumento
            sigma_prime = self.activation_derivative(a_j)

            # Término gamma auxiliar
            gamma = sigma_prime * p

            # Ecuación adjunta discreta
            # p_j = p_{j+1} + delta_t * gamma * K
            p = p + self.delta_t * (gamma @ layer.weight)
            p_list[j] = p

        p_traj = torch.stack(p_list, dim=0)
        return p_traj, epsilon

    @torch.no_grad()
    def calculate_gradients(
        self,
        y_traj: torch.Tensor,
        a_list,
        p_traj: torch.Tensor,
        epsilon: torch.Tensor,
    ):
        """
        Calcula los gradientes de los parámetros (K, beta, W, b) usando
        las trayectorias de estado 'y' y co-estado 'p'.

        Según PMP, el gradiente es la derivada del Hamiltoniano con respecto al control.
        """
        grad_list_K = []
        grad_list_beta = []

        B = y_traj.size(1)  # Batch size

        # Gradientes para las capas internas (Controles K y beta)
        for j in range(self.N):
            y = y_traj[j]
            a = a_list[j]
            p = p_traj[j + 1]  # Usamos p_{j+1} según la discretización

            gamma = self.activation_derivative(a) * p

            # dH/dK
            grad_K = (1 / B) * self.delta_t * (gamma.T @ y)
            # dH/dbeta
            grad_beta = (1 / B) * self.delta_t * gamma.sum(dim=0)

            grad_list_K.append(grad_K)
            grad_list_beta.append(grad_beta)

        # Gradientes para la capa final (Parámetros W y b de costo terminal)
        yN = y_traj[-1]
        grad_W = (1 / B) * epsilon.T @ yN
        grad_b = (1 / B) * epsilon.sum(dim=0)

        return grad_list_K, grad_list_beta, grad_W, grad_b

    @torch.no_grad()
    def compute_loss(
        self, y0: torch.Tensor, c: torch.Tensor, average_loss: bool = True
    ):
        """Calcula la pérdida MSE (Mean Squared Error)."""
        y = self.forward(y0)
        c = c.view(-1, 1)
        resid = y - c
        loss = 0.5 * (resid**2).mean() if average_loss else 0.5 * (resid**2).sum()
        return loss.item()

    # --- Funciones auxiliares para Backtracking ---

    @torch.no_grad()
    def _pack_params(self):
        """Empaqueta parámetros actuales y crea un backup para restaurar si el paso se rechaza."""
        params = []
        backup = []
        for layer in self.inner_layers:
            params += [layer.weight, layer.bias]
        params += [self.last_layer.weight, self.last_layer.bias]
        for p in params:
            backup.append(p.detach().clone())
        return params, backup

    @torch.no_grad()
    def _apply_step(self, params, step):
        """Aplica un paso de gradiente a los parámetros."""
        for p, s in zip(params, step):
            p.data.add_(s)

    @torch.no_grad()
    def _restore(self, params, backup):
        """Restaura los parámetros desde el backup."""
        for p, b in zip(params, backup):
            p.data.copy_(b)

    @torch.no_grad()
    def update_params_backtracking_global(
        self,
        y0,
        c,
        grad_list_K,
        grad_list_beta,
        grad_W,
        grad_b,
        L: float,
        rho: float,
        rbar: float,
        max_inner: int = 50,
    ):
        """
        Realiza la actualización de parámetros usando Line Search (Backtracking).
        Busca un paso de aprendizaje 1/L que satisfaga la condición de Armijo.
        """
        params, backup = self._pack_params()

        # Aplanar lista de gradientes
        grads = []
        for j in range(self.N):
            grads += [
                grad_list_K[j],
                grad_list_beta[j].view_as(self.inner_layers[j].bias),
            ]
        grads += [grad_W, grad_b.view_as(self.last_layer.bias)]

        # Pérdida actual
        phi = self.compute_loss(y0, c, average_loss=True)

        # Bucle de búsqueda de línea
        for _ in range(max_inner):
            # Paso propuesto: - (1/L) * gradiente
            step = [-(1.0 / L) * g for g in grads]
            self._apply_step(params, step)

            # Calcular nueva pérdida
            phi_t = self.compute_loss(y0, c, average_loss=True)

            # Condición de descenso suficiente (Armijo)
            inner = 0.0
            quad = 0.0
            for g, s in zip(grads, step):
                inner += (g * s).sum().item()
                quad += (s * s).sum().item()
            rhs = phi + inner + 0.5 * L * quad

            if phi_t <= rhs:
                # Paso aceptado
                return rho * L, True  # Reducimos L ligeramente para el futuro
            else:
                # Paso rechazado, restaurar y aumentar L (disminuir paso)
                self._restore(params, backup)
                L = rbar * L

        return L, False

    @torch.no_grad()
    def train_step_backtracking(
        self,
        y0,
        c,
        average_loss: bool = True,
        L: float = 1.0,
        rho: float = 0.5,
        rbar: float = 2.0,
    ):
        """
        Ejecuta un paso completo de entrenamiento:
        1. Forward (Trayectoria y)
        2. Backward (Trayectoria p)
        3. Gradientes (PMP)
        4. Actualización (Backtracking)
        """
        # 1. Calcular trayectoria de estado
        y_traj, a_list = self.calculate_y_trayectory(y0)
        phi = self.compute_loss(y0, c, average_loss)

        # 2. Calcular variables adjuntas
        p_traj, epsilon = self.calculate_p_trayectory(y_traj, a_list, c)

        # 3. Calcular gradientes
        grad_list_K, grad_list_beta, grad_W, grad_b = self.calculate_gradients(
            y_traj, a_list, p_traj, epsilon
        )

        # 4. Actualizar parámetros
        L, accepted = self.update_params_backtracking_global(
            y0, c, grad_list_K, grad_list_beta, grad_W, grad_b, L, rho, rbar
        )

        return phi, L, accepted
