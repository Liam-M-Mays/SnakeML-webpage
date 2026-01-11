# Snake Reinforcement Learning

An interactive web application for training and visualizing reinforcement learning agents playing Snake. Watch AI learn in real-time with adjustable hyperparameters.

## Features

- **Multiple RL Algorithms**: Choose between DQN (Deep Q-Network) and PPO (Proximal Policy Optimization)
- - **Real-time Training Visualization**: Watch the agent learn and improve
  - - **Hyperparameter Control**: Adjust learning rate, discount factor, and other parameters on the fly
    - - **Training Metrics**: Track scores, rewards, and learning progress
      - - **Interactive Gameplay**: Play Snake yourself or let the AI take over
       
        - ## Tech Stack
       
        - **Frontend:**
        - - React + Vite
          - - Real-time game rendering
           
            - **Backend:**
            - - Python / Flask
              - - PyTorch for neural networks
                - - Custom RL environment
                 
                  - ## Project Structure
                 
                  - ```
                    ├── backend/          # Python RL training server
                    │   ├── models/       # Neural network architectures
                    │   ├── agents/       # DQN, PPO implementations
                    │   └── environment/  # Snake game environment
                    ├── src/              # React frontend
                    │   ├── components/   # UI components
                    │   └── hooks/        # Custom React hooks
                    └── environment.yml   # Conda environment
                    ```

                    ## Getting Started

                    ### Backend Setup
                    ```bash
                    cd backend
                    conda env create -f ../environment.yml
                    conda activate snake-rl
                    python app.py
                    ```

                    ### Frontend Setup
                    ```bash
                    npm install
                    npm run dev
                    ```

                    ## Roadmap

                    - [ ] Add more RL algorithms (A2C, SAC)
                    - [ ] - [ ] Implement more games (Tetris, Flappy Bird)
                    - [ ] - [ ] Add model saving/loading
                    - [ ] - [ ] Improve visualization dashboard
                    - [ ] - [ ] Deploy to web
