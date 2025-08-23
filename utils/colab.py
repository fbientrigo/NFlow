# File that references functions in order to run specific tasks
# of training in Google Colab.
import subprocess
import logging

logger = logging.getLogger(__name__)

def get_gpu_info():
    """
    Executes nvidia-smi command using subprocess to get GPU information.
    Returns a list of strings, each representing a line of the output.
    Returns an empty list and logs a warning if nvidia-smi is not found or fails.
    """
    try:
        # Use subprocess to run the command
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, check=True)
        gpu_info = result.stdout.strip().split('\n')
        return gpu_info
    except FileNotFoundError:
        logger.warning("nvidia-smi command not found. Is NVIDIA driver installed?")
        return []
    except subprocess.CalledProcessError as e:
        logger.error(f"Error executing nvidia-smi: {e}")
        logger.error(f"Stderr: {e.stderr}")
        return []
    except Exception as e:
        logger.error(f"An unexpected error occurred while getting GPU info: {e}")
        return []


def welcoming():
    """
    Prints a welcoming message and displays GPU information if available.
    """
    print("Welcome to NFlow Collaboratory!")
    print("Setting up your environment...")

    gpu_info = get_gpu_info()
    if gpu_info:
        print("\nGPU Information:")
        for line in gpu_info:
            print(line)
    else:
        print("\nNo GPU information available.")

    print("\nReady to start!")

if __name__ == '__main__':
    # Example usage
    welcoming()