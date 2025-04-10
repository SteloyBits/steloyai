import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser(description="AI Unified Platform")
    parser.add_argument("--interface", type=str, choices=["gradio", "streamlit"], default="gradio", 
                        help="Interface to use (gradio or streamlit)")
    parser.add_argument("--share", action="store_true", help="Share the application publicly (for Gradio)")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    args = parser.parse_args()
    
    if args.interface == "gradio":
        from interface.gradio_app import GradioInterface
        app = GradioInterface()
        app.launch(share=args.share, debug=args.debug)
    elif args.interface == "streamlit":
        print("To run the Streamlit interface, use: streamlit run src/interface/streamlit_app.py")
        sys.exit(0)

if __name__ == "__main__":
    main()
