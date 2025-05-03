/*  */import styled, { createGlobalStyle } from 'styled-components';

export const GlobalStyle = createGlobalStyle`
    body {
        font-family: 'Cascadia Code', monospace;
        background-color: #f5f5f5;
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
`;

export const Container = styled.div`
    display: grid;
    grid-template-columns: 2fr 1fr;
    grid-template-rows: auto auto auto auto;
    gap: 20px;
    width: 100%;
    height: 100vh;
    margin: 0;
    padding: 20px;
    box-sizing: border-box;
`;

export const HeaderContainer = styled.div`
    grid-column: span 2;
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0 0 20px 0;
    border-bottom: 1px solid #d0d0d0;
`;

export const HeaderLogo = styled.div`
    display: flex;
    align-items: center;
    gap: 10px;
`;

export const Header = styled.h1`
    font-size: 32px;
    font-weight: 700;
    color: #ffffff;
    margin: 0;
    background-color: #3e5caa;
    padding: 8px 16px;
    border-radius: 0;
`;

export const HeaderNav = styled.div`
    display: flex;
    gap: 10px;
`;

export const Tagline = styled.p`
    grid-column: span 2;
    font-size: 18px;
    color: #3e5caa;
    text-align: center;
    margin: 0 0 20px 0;
`;

export const MapContainer = styled.div`
    grid-column: 1 / 2;
    background-color: #e1e8f7;
    border: 2px solid #3e5caa;
    border-radius: 12px;
    width: 100%;
    height: 100vh;
    display: flex;
    flex-direction: column,
    justify-content: stretch;
    align-items: stretch;
    position: relative;
    overflow: hidden;

    iframe {
        width: 100%;
        height: 100vh;
        border: none;
        display: block;
    }
`;

export const LoadingMessage = styled.p`
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    font-size: 20px;
    font-weight: bold;
    color: #3e5caa;
    background-color: rgba(255, 255, 255, 0.8);
    padding: 10px 20px;
    border-radius: 8px;
`;

export const GraphLoadingMessage = styled.div`
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    font-size: 20px;
    font-weight: bold;
    color: #3e5caa;
    background-color: rgba(255, 255, 255, 0.8);
    padding: 10px 20px;
    border-radius: 8px;
    display: flex;
    justify-content: center;
    align-items: center;
`;



export const PromptContainer = styled.div`
    grid-column: 1 / 2;
    padding: 20px;
    background-color: #ffffff;
    border-radius: 15px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    display: flex;
    flex-direction: column;
    gap: 20px;
`;

export const TextArea = styled.textarea`
    width: 100%;
    height: 80px;
    padding: 12px;
    border: 2px solid #3e5caa;
    border-radius: 10px;
    outline: none;
    font-size: 16px;
    font-family: 'Cascadia Code', monospace;
    resize: none;
`;

export const Button = styled.button`
    padding: 10px 20px;
    background-color: #3e5caa;
    color: #fff;
    border: none;
    border-radius: 8px;
    cursor: pointer;
    font-size: 16px;
    font-weight: 600;
    font-family: 'Cascadia Code', monospace;
    transition: background 0.3s ease-in-out;
    
    &:hover {
        background-color: #2c4885;
    }
    
    &.amenity-score-button {
        align-self: center;
        min-width: 180px;
    }
`;

export const GenerateButton = styled(Button)`
    width: 100%;
    padding: 15px;
    font-size: 18px;
    text-transform: none;
    max-width: 300px;
    align-self: center;
`;


export const FilterContainer = styled.div`
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 15px;
    
    @media (min-width: 768px) {
        grid-template-columns: repeat(4, 1fr);
    }
    
    div {
        display: flex;
        align-items: center;
        gap: 10px;
    }
`;


export const AnalyticsTitle = styled.h2`
    font-size: 24px;
    color: #3e5caa;
    margin: 0;
`;


